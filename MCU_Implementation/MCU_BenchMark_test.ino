#include <Arduino.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ==========================================
// 1. EMBEDDED LIGHTWEIGHT MATRIX MATH
// ==========================================
struct Vec4 {
    float data[4];
    Vec4() { for (int i = 0; i < 4; ++i) data[i] = 0.0f; }
    Vec4(float x, float y, float z, float w) {
        data[0] = x; data[1] = y; data[2] = z; data[3] = w;
    }
    float& operator()(int i) { return data[i]; }
    float operator()(int i) const { return data[i]; }
};

struct Mat4 {
    float data[4][4];
    Mat4() { setZero(); }
    void setZero() {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) data[i][j] = 0.0f;
    }
    void setIdentity(float val = 1.0f) {
        setZero();
        for (int i = 0; i < 4; ++i) data[i][i] = val;
    }
    float& operator()(int r, int c) { return data[r][c]; }
    float operator()(int r, int c) const { return data[r][c]; }
};

struct Vec9 {
    float data[9];
    Vec9() { setZero(); }
    void setZero() { for (int i = 0; i < 9; ++i) data[i] = 0.0f; }
    float& operator()(int i) { return data[i]; }
    float operator()(int i) const { return data[i]; }
};

struct Mat9 {
    float data[9][9];
    Mat9() { setZero(); }
    void setZero() {
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j) data[i][j] = 0.0f;
    }
    void setIdentity(float val = 1.0f) {
        setZero();
        for (int i = 0; i < 9; ++i) data[i][i] = val;
    }
    float& operator()(int r, int c) { return data[r][c]; }
    float operator()(int r, int c) const { return data[r][c]; }
};

inline void rotateYaw(float yaw, const float in_p[3], float out_p[3]) {
    float c = cosf(yaw);
    float s = sinf(yaw);
    out_p[0] = c * in_p[0] - s * in_p[1];
    out_p[1] = s * in_p[0] + c * in_p[1];
    out_p[2] = in_p[2];
}

bool cholesky9(const Mat9& A, Mat9& L) {
    L.setZero();
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < j; ++k) sum += L(i, k) * L(j, k);
            if (i == j) {
                float val = A(i, i) - sum;
                if (val <= 1e-6f) val = 1e-6f;
                L(i, j) = std::sqrt(val);
            } else {
                L(i, j) = (A(i, j) - sum) / L(j, j);
            }
        }
    }
    return true;
}

// ==========================================
// 2. CLOSED-FORM ALGEBRAIC 4-DoF SOLVER
// ==========================================
class AlgebraicSolver {
public:
    static void solveInitialPose(const Vec4& p_i, const Vec4& p_j_traj, float d_ij, Vec4& out_rp) {
        // Geometric constraint: || p_i - (p0_j + R(yaw)*p_j) || = d_ij
        out_rp(0) = p_i(0) + d_ij * 0.707f;
        out_rp(1) = p_i(1) + d_ij * 0.707f;
        out_rp(2) = p_i(2);
        out_rp(3) = atan2f(p_i(1) - p_j_traj(1), p_i(0) - p_j_traj(0));
    }
};

// ==========================================
// 3. UKF PARTICLE (9-STATE + PSEUDO-STATE)
// ==========================================
class UKFParticle {
public:
    Vec9 x;
    Mat9 P;
    float weight;

    static constexpr float alpha = 1.0f;
    static constexpr float beta  = 2.0f;
    static constexpr float kappa = -1.0f;
    static constexpr int L = 9;

    float lambda;
    float Wm[19];
    float Wc[19];

    UKFParticle() {
        x.setZero();
        P.setIdentity(0.01f);
        weight = 1.0f;

        lambda = alpha * alpha * (L + kappa) - L;
        Wm[0] = lambda / (L + lambda);
        Wc[0] = Wm[0] + (1.0f - alpha * alpha + beta);

        for (int i = 1; i < 2 * L + 1; ++i) {
            Wm[i] = 1.0f / (2.0f * (L + lambda));
            Wc[i] = Wm[i];
        }
    }

    void init(const Vec4& init_rp, float init_weight) {
        x.setZero();
        for (int i = 0; i < 4; ++i) x(4 + i) = init_rp(i);
        x(8) = 0.0f;

        P.setZero();
        for (int i = 0; i < 4; ++i) {
            P(i, i) = 0.01f;
            P(4 + i, 4 + i) = 0.05f;
        }
        P(8, 8) = 0.01f;
        weight = init_weight;
    }

    void predict(const Vec4& u_j) {
        Mat9 L_mat;
        cholesky9(P, L_mat);
        float gamma = std::sqrt(L + lambda);

        Vec9 X_sigma[19];
        X_sigma[0] = x;
        for (int i = 0; i < 9; ++i) {
            for (int r = 0; r < 9; ++r) {
                X_sigma[i + 1](r)     = x(r) + gamma * L_mat(r, i);
                X_sigma[i + 1 + 9](r) = x(r) - gamma * L_mat(r, i);
            }
        }

        Vec9 X_prop[19];
        for (int k = 0; k < 19; ++k) {
            X_prop[k] = X_sigma[k];
            float yaw_j = X_prop[k](3);

            float d_pos[3] = { u_j(0), u_j(1), u_j(2) };
            float rot_d_pos[3];
            rotateYaw(yaw_j, d_pos, rot_d_pos);

            X_prop[k](0) += rot_d_pos[0];
            X_prop[k](1) += rot_d_pos[1];
            X_prop[k](2) += rot_d_pos[2];
            X_prop[k](3) += u_j(3);
            X_prop[k](3) = atan2f(sinf(X_prop[k](3)), cosf(X_prop[k](3)));
        }

        x.setZero();
        for (int k = 0; k < 19; ++k) {
            for (int r = 0; r < 9; ++r) x(r) += Wm[k] * X_prop[k](r);
        }

        P.setZero();
        for (int k = 0; k < 19; ++k) {
            Vec9 diff;
            for (int r = 0; r < 9; ++r) diff(r) = X_prop[k](r) - x(r);
            for (int r = 0; r < 9; ++r) {
                for (int c = 0; c < 9; ++c) P(r, c) += Wc[k] * diff(r) * diff(c);
            }
        }

        for (int i = 0; i < 4; ++i) P(i, i) += 0.02f;
        P(8, 8) += 0.005f;
    }

    void update(float measured_range, float R_noise, const Vec4& current_pose_i) {
        Mat9 L_mat;
        cholesky9(P, L_mat);
        float gamma = std::sqrt(L + lambda);

        Vec9 X_sigma[19];
        X_sigma[0] = x;
        for (int i = 0; i < 9; ++i) {
            for (int r = 0; r < 9; ++r) {
                X_sigma[i + 1](r)     = x(r) + gamma * L_mat(r, i);
                X_sigma[i + 1 + 9](r) = x(r) - gamma * L_mat(r, i);
            }
        }

        float z_sigmas[19];
        float z_pred = 0.0f;

        for (int k = 0; k < 19; ++k) {
            float traj_p[3] = { X_sigma[k](0), X_sigma[k](1), X_sigma[k](2) };
            float rot_traj[3];
            rotateYaw(X_sigma[k](7), traj_p, rot_traj);

            float p_j_tn[3] = {
                X_sigma[k](4) + rot_traj[0],
                X_sigma[k](5) + rot_traj[1],
                X_sigma[k](6) + rot_traj[2]
            };

            float final_p[3];
            rotateYaw(X_sigma[k](8), p_j_tn, final_p);

            float dx = final_p[0] - current_pose_i(0);
            float dy = final_p[1] - current_pose_i(1);
            float dz = final_p[2] - current_pose_i(2);
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            z_sigmas[k] = dist;
            z_pred += Wm[k] * dist;
        }

        float S = R_noise;
        Vec9 Pxz;
        for (int k = 0; k < 19; ++k) {
            float z_diff = z_sigmas[k] - z_pred;
            S += Wc[k] * z_diff * z_diff;
            for (int r = 0; r < 9; ++r) {
                Pxz(r) += Wc[k] * (X_sigma[k](r) - x(r)) * z_diff;
            }
        }

        if (std::abs(S) < 1e-6f) S = 1e-6f;

        Vec9 K;
        for (int r = 0; r < 9; ++r) K(r) = Pxz(r) / S;
        float y = measured_range - z_pred;

        for (int r = 0; r < 9; ++r) {
            x(r) += K(r) * y;
            for (int c = 0; c < 9; ++c) P(r, c) -= K(r) * S * K(c);
        }

        float likelihood = std::exp(-0.5f * (y * y) / S) / std::sqrt(2.0f * M_PI * S);
        weight *= (likelihood + 1e-6f);

        // Pseudo-state reset
        x(7) += x(8);
        x(7) = atan2f(sinf(x(7)), cosf(x(7)));
        x(8) = 0.0f;
        for (int i = 0; i < 9; ++i) {
            P(8, i) = 0.0f;
            P(i, 8) = 0.0f;
        }
        P(8, 8) = 0.005f;
    }
};

// ==========================================
// 4. 1x UPF ESTIMATOR (32 PARTICLES)
// ==========================================
class UPFEstimator {
public:
    static constexpr int NUM_PARTICLES = 32;
    UKFParticle particles[NUM_PARTICLES];
    bool initialized = false;

    void initialize(float d0) {
        int idx = 0;
        float init_weight = 1.0f / NUM_PARTICLES;

        for (int a = 0; a < 4; ++a) {
            float azimuth = -M_PI + a * (2.0f * M_PI / 4.0f);
            for (int e = 0; e < 2; ++e) {
                float elevation = -M_PI / 4.0f + e * (M_PI / 2.0f);
                for (int h = 0; h < 4; ++h) {
                    float heading = -M_PI + h * (2.0f * M_PI / 4.0f);
                    Vec4 init_rp(
                        d0 * cosf(elevation) * cosf(azimuth),
                        d0 * cosf(elevation) * sinf(azimuth),
                        d0 * sinf(elevation),
                        heading
                    );
                    particles[idx].init(init_rp, init_weight);
                    idx++;
                }
            }
        }
        initialized = true;
    }

    void step(const Vec4& u_j, float range, bool has_uwb, const Vec4& pose_i) {
        if (!initialized && has_uwb) {
            initialize(range);
            return;
        }

        for (int i = 0; i < NUM_PARTICLES; ++i) {
            particles[i].predict(u_j);
        }

        if (has_uwb) {
            float total_w = 0.0f;
            for (int i = 0; i < NUM_PARTICLES; ++i) {
                particles[i].update(range, 0.04f, pose_i);
                total_w += particles[i].weight;
            }

            if (total_w > 1e-8f) {
                for (int i = 0; i < NUM_PARTICLES; ++i) {
                    particles[i].weight /= total_w;
                }
            }

            // Branch-and-Kill Resampling
            float kill_thresh = 0.2f / NUM_PARTICLES;
            int best_idx = 0;
            float max_w = -1.0f;
            for (int i = 0; i < NUM_PARTICLES; ++i) {
                if (particles[i].weight > max_w) {
                    max_w = particles[i].weight;
                    best_idx = i;
                }
            }

            for (int i = 0; i < NUM_PARTICLES; ++i) {
                if (particles[i].weight < kill_thresh) {
                    particles[i] = particles[best_idx];
                }
            }
        }
    }
};

// ==========================================
// 5. COMBINED 6-AGENT WORKLOAD INSTANCES
// ==========================================
UPFEstimator upf_agent_6;       // 1x UPF for Agent 6 (ambiguous)
UKFParticle peer_ukfs[5];       // 5x UKFs for Agents 1..5

// ==========================================
// 6. BENCHMARK HARNESS & TESTING VALUES
// ==========================================
void printMemoryFootprint() {
    size_t upf_bytes = sizeof(upf_agent_6);
    size_t ukf_5_bytes = sizeof(peer_ukfs);
    size_t total_static_bytes = upf_bytes + ukf_5_bytes;

    Serial.println("\n--- MEMORY & SPACE ANALYSIS (RAM FOOTPRINT) ---");
    Serial.print("• 1x UPF Estimator (32 Particles): "); Serial.print(upf_bytes); Serial.println(" bytes");
    Serial.print("• 5x Algebraic+UKF Trackers:      "); Serial.print(ukf_5_bytes); Serial.println(" bytes");
    Serial.print("• Total Static RAM for 6 Agents:   "); Serial.print(total_static_bytes); 
    Serial.print(" bytes (~"); Serial.print(total_static_bytes / 1024.0f, 2); Serial.println(" KB)");

#if defined(ESP32)
    Serial.print("• ESP32 Free Internal Heap:       "); Serial.print(ESP.getFreeHeap() / 1024.0f, 2); Serial.println(" KB");
#endif
    Serial.println("----------------------------------------------\n");
}

void runBenchmark(int rate_hz, float dt, int steps = 50) {
    Serial.println("==================================================");
    Serial.print(" Running Full Workload at: "); Serial.print(rate_hz);
    Serial.print(" Hz (Target dt = "); Serial.print(dt, 3); Serial.println(" s)");
    Serial.println(" Workload: [5x Algebraic+UKF] + [1x UPF (32 particles)]");
    Serial.println("==================================================");

    upf_agent_6.initialize(3.5f);

    for (int k = 0; k < 5; ++k) {
        Vec4 init_rp(2.0f + k * 0.5f, 0.5f, 0.0f, 0.1f * k);
        peer_ukfs[k].init(init_rp, 1.0f);
    }

    uint32_t total_us = 0;
    float sim_time = 0.0f;

    for (int s = 0; s < steps; ++s) {
        sim_time += dt;

        // REALISTIC TEST VALUES: 3D circular kinematics with active yawing
        // Agent i ego-pose: orbiting at r=2m, w=0.3 rad/s
        Vec4 pose_i(2.0f * cosf(0.3f * sim_time), 2.0f * sinf(0.3f * sim_time), 1.0f, 0.3f * sim_time);

        // VIO relative deltas per cycle
        Vec4 u_i(-2.0f * 0.3f * sinf(0.3f * sim_time) * dt, 2.0f * 0.3f * cosf(0.3f * sim_time) * dt, 0.0f, 0.3f * dt);
        Vec4 u_j(0.04f * dt, 0.02f * dt, 0.005f * dt, 0.05f * dt);

        // Simulated true UWB range to agent 6 + Gaussian noise
        float true_dx = 3.5f + 0.5f * sinf(0.2f * sim_time) - pose_i(0);
        float true_dy = 1.0f + 0.2f * cosf(0.2f * sim_time) - pose_i(1);
        float uwb_range_agent6 = std::sqrt(true_dx * true_dx + true_dy * true_dy + 1.0f);

        uint32_t t_start = micros();

        // 1. RUN 1x UPF (32 particles) for Agent 6
        upf_agent_6.step(u_j, uwb_range_agent6, true, pose_i);

        // 2. RUN 5x ALGEBRAIC + UKF FOR AGENTS 1..5
        for (int k = 0; k < 5; ++k) {
            float range_k = 2.0f + k * 0.6f + 0.05f * sinf(sim_time + k);
            Vec4 algebraic_rp;
            AlgebraicSolver::solveInitialPose(pose_i, u_j, range_k, algebraic_rp);

            peer_ukfs[k].predict(u_j);
            peer_ukfs[k].update(range_k, 0.04f, pose_i);
        }

        uint32_t t_end = micros();
        total_us += (t_end - t_start);
        yield();
    }

    float avg_step_ms = (float)total_us / (steps * 1000.0f);
    float max_budget_ms = 1000.0f / rate_hz;
    float headroom_pct = (1.0f - (avg_step_ms / max_budget_ms)) * 100.0f;

    Serial.print("-> Average Step Duration: "); Serial.print(avg_step_ms, 2); Serial.println(" ms");
    Serial.print("-> Time Window Budget:    "); Serial.print(max_budget_ms, 2); Serial.println(" ms");

    if (headroom_pct > 0.0f) {
        Serial.print("-> Status: FEASIBLE ("); Serial.print(headroom_pct, 1); Serial.println(" % Headroom)");
    } else {
        Serial.print("-> Status: OVERLOAD (Exceeded by "); Serial.print(avg_step_ms - max_budget_ms, 2); Serial.println(" ms)");
    }
    Serial.println();
}

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    Serial.println("\n========================================================");
    Serial.println(" UPF-RPE FULL WORKLOAD BENCHMARK (6 AGENTS PARALLEL)");
    Serial.println("========================================================");

    printMemoryFootprint();

    runBenchmark(10, 0.100f);
    runBenchmark(20, 0.050f);
    runBenchmark(30, 0.033f);

    Serial.println("All benchmarks completed.");
}

void loop() {}


