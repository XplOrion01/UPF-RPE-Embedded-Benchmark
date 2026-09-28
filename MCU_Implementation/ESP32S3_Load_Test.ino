/*
 * ESP32-S3 UPF-RPE RESOURCE BENCHMARK
 *
 * Purpose:
 *   Determine whether an ESP32-S3 can support the intended workload:
 *
 *       5 x Algebraic 4DoF estimators
 *       1 x UPF estimator
 *       + IMU workload
 *       + WiFi workload
 *       + UART telemetry
 *
 * This is a RESOURCE / TIMING benchmark.
 *
 * IMPORTANT:
 * This benchmark is NOT yet the final line-by-line C++ port
 * of Yuri sir's Python UPF-RPE implementation.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <math.h>
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// ============================================================
// CONFIGURATION
// ============================================================

#define NUM_ALG_ESTIMATORS 5
#define NUM_UPF_PARTICLES 32

#define NUM_STEPS 200

#define ENABLE_WIFI_TEST       1
#define ENABLE_IMU_TEST        1
#define ENABLE_UART_TEST       1

#define ESTIMATOR_CORE 1
#define SYSTEM_CORE    0

#define TELEMETRY_BAUD 115200

// ============================================================
// BASIC TYPES
// ============================================================

struct Vec4 {
    float v[4];

    float &operator()(int i) {
        return v[i];
    }

    const float &operator()(int i) const {
        return v[i];
    }
};

struct Mat4 {
    float m[4][4];

    float &operator()(int r, int c) {
        return m[r][c];
    }

    const float &operator()(int r, int c) const {
        return m[r][c];
    }
};

struct Vec9 {
    float v[9];

    float &operator()(int i) {
        return v[i];
    }

    const float &operator()(int i) const {
        return v[i];
    }
};

struct Mat9 {
    float m[9][9];

    float &operator()(int r, int c) {
        return m[r][c];
    }

    const float &operator()(int r, int c) const {
        return m[r][c];
    }
};

// ============================================================
// GLOBALS
// ============================================================

volatile bool estimatorTaskRunning = false;
volatile bool systemTaskRunning = false;

volatile uint32_t estimatorSteps = 0;
volatile uint32_t systemSteps = 0;

volatile uint32_t deadlineMisses = 0;

TaskHandle_t estimatorTaskHandle = nullptr;
TaskHandle_t systemTaskHandle = nullptr;

// ============================================================
// MATH UTILITIES
// ============================================================

static inline float limitAngle(float a)
{
    while (a > PI)
        a -= 2.0f * PI;

    while (a <= -PI)
        a += 2.0f * PI;

    return a;
}

static inline Mat4 identity4()
{
    Mat4 A{};

    for (int i = 0; i < 4; i++)
        A(i, i) = 1.0f;

    return A;
}

static inline Mat4 get4DRotation(float h)
{
    Mat4 R = identity4();

    float c = cosf(h);
    float s = sinf(h);

    R(0,0) = c;
    R(0,1) = -s;
    R(1,0) = s;
    R(1,1) = c;

    return R;
}

static inline Vec4 matVec4(
    const Mat4 &A,
    const Vec4 &x)
{
    Vec4 y{};

    for (int i = 0; i < 4; i++)
    {
        y(i) = 0.0f;

        for (int j = 0; j < 4; j++)
            y(i) += A(i,j) * x(j);
    }

    return y;
}

static inline float norm3(
    float x,
    float y,
    float z)
{
    return sqrtf(
        x*x +
        y*y +
        z*z
    );
}

// ============================================================
// ALGEBRAIC 4DOF ESTIMATOR
// ============================================================
//
// Structure follows Yuri sir's AlgebraicMethod4DoF:
//   - 4 DoF state
//   - 10 measurement horizon
//   - host odometry
//   - connected-agent odometry
//   - UWB range
//   - algebraic least-squares solve
//
// ============================================================

class AlgebraicEstimator
{
public:

    static constexpr int HORIZON = 10;

    float d0;

    Vec4 hostOdom[HORIZON];
    Vec4 connectedOdom[HORIZON];

    float eps[HORIZON];
    float distance[HORIZON];

    int count;

    Vec4 x_ca_0_alg;
    Vec4 x_ca_r_alg;

    AlgebraicEstimator()
    {
        reset(3.5f);
    }

    void reset(float initialDistance)
    {
        d0 = initialDistance;

        count = 1;

        for (int i = 0; i < HORIZON; i++)
        {
            hostOdom[i] = {};
            connectedOdom[i] = {};
            eps[i] = 0.0f;
            distance[i] = 0.0f;
        }

        distance[0] = d0;

        x_ca_0_alg = {};
        x_ca_r_alg = {};
    }

    void update(
        float d,
        const Vec4 &dxHost,
        const Vec4 &dxConnected)
    {
        Vec4 hostPrevious =
            hostOdom[count - 1];

        Vec4 connectedPrevious =
            connectedOdom[count - 1];

        Mat4 Ch =
            get4DRotation(hostPrevious(3));

        Mat4 Cc =
            get4DRotation(connectedPrevious(3));

        Vec4 hostCurrent =
            hostPrevious;

        Vec4 connectedCurrent =
            connectedPrevious;

        Vec4 dh =
            matVec4(Ch, dxHost);

        Vec4 dc =
            matVec4(Cc, dxConnected);

        for (int i = 0; i < 4; i++)
        {
            hostCurrent(i) += dh(i);
            connectedCurrent(i) += dc(i);
        }

        float e =
            0.5f *
            (
                d0*d0
                +
                hostCurrent(0)*hostCurrent(0)
                +
                hostCurrent(1)*hostCurrent(1)
                +
                hostCurrent(2)*hostCurrent(2)
                +
                connectedCurrent(0)*connectedCurrent(0)
                +
                connectedCurrent(1)*connectedCurrent(1)
                +
                connectedCurrent(2)*connectedCurrent(2)
                -
                d*d
            );

        if (count < HORIZON)
        {
            hostOdom[count] = hostCurrent;
            connectedOdom[count] = connectedCurrent;

            eps[count] = e;
            distance[count] = d;

            count++;
        }
        else
        {
            for (int i = 1; i < HORIZON; i++)
            {
                hostOdom[i-1] =
                    hostOdom[i];

                connectedOdom[i-1] =
                    connectedOdom[i];

                eps[i-1] =
                    eps[i];

                distance[i-1] =
                    distance[i];
            }

            hostOdom[HORIZON-1] =
                hostCurrent;

            connectedOdom[HORIZON-1] =
                connectedCurrent;

            eps[HORIZON-1] =
                e;

            distance[HORIZON-1] =
                d;
        }

        if (count >= HORIZON)
            solve();
    }

    void solve()
    {
        /*
         * Lightweight embedded algebraic solve.
         *
         * The FINAL version will replace this section with
         * the complete numerical translation of Yuri sir's
         * AlgebraicMethod4DoF.find_relative_pose().
         */

        float sx = 0.0f;
        float sy = 0.0f;
        float sz = 0.0f;

        for (int i = 1; i < HORIZON; i++)
        {
            sx += connectedOdom[i](0)
                - hostOdom[i](0);

            sy += connectedOdom[i](1)
                - hostOdom[i](1);

            sz += connectedOdom[i](2)
                - hostOdom[i](2);
        }

        float inv =
            1.0f / float(HORIZON - 1);

        x_ca_0_alg(0) = sx * inv;
        x_ca_0_alg(1) = sy * inv;
        x_ca_0_alg(2) = sz * inv;

        x_ca_0_alg(3) =
            limitAngle(
                connectedOdom[HORIZON-1](3)
                -
                hostOdom[HORIZON-1](3)
            );

        Mat4 Ch =
            get4DRotation(
                hostOdom[HORIZON-1](3)
            );

        Vec4 difference{};

        for (int i = 0; i < 4; i++)
        {
            difference(i) =
                x_ca_0_alg(i)
                +
                connectedOdom[HORIZON-1](i)
                -
                hostOdom[HORIZON-1](i);
        }

        Vec4 result =
            matVec4(Ch, difference);

        x_ca_r_alg =
            result;

        x_ca_r_alg(3) =
            limitAngle(
                x_ca_0_alg(3)
                +
                connectedOdom[HORIZON-1](3)
                -
                hostOdom[HORIZON-1](3)
            );
    }
};

// ============================================================
// SIMPLIFIED 9D UKF PARTICLE
// ============================================================

class UKFParticle
{
public:

    Vec9 x;
    Mat9 P;

    float weight;

    float sigmaUwb;

    static constexpr int N_SIGMA = 19;

    Vec9 sigmaPoints[N_SIGMA];

    UKFParticle()
    {
        reset();
    }

    void reset()
    {
        for (int i = 0; i < 9; i++)
        {
            x(i) = 0.0f;

            for (int j = 0; j < 9; j++)
                P(i,j) =
                    (i == j)
                    ? 0.1f
                    : 0.0f;
        }

        weight = 1.0f;

        sigmaUwb =
            0.1f;
    }

    void predict(
        const Vec4 &dx)
    {
        /*
         * State:
         *
         * 0 = range
         * 1 = azimuth
         * 2 = altitude
         * 3 = connected heading
         * 4 = connected x odom
         * 5 = connected y odom
         * 6 = connected z odom
         * 7 = connected heading odom
         * 8 = host heading drift
         */

        x(4) += dx(0);
        x(5) += dx(1);
        x(6) += dx(2);
        x(7) =
            limitAngle(
                x(7) + dx(3)
            );

        generateSigmaPoints();
    }

    float predictedRange()
    {
        float r =
            fabsf(x(0));

        if (r < 0.001f)
            r = 0.001f;

        return r;
    }

    void update(float measuredRange)
    {
        float predicted =
            predictedRange();

        float error =
            measuredRange - predicted;

        float variance =
            sigmaUwb*sigmaUwb + 0.01f;

        float likelihood =
            expf(
                -0.5f *
                error*error /
                variance
            );

        if (likelihood < 1e-20f)
            likelihood = 1e-20f;

        weight *= likelihood;

        /*
         * Simple scalar correction.
         */
        float K =
            P(0,0) /
            (P(0,0) + variance);

        x(0) +=
            K * error;

        P(0,0) =
            (1.0f - K) *
            P(0,0);
    }

    void generateSigmaPoints()
    {
        /*
         * 9-dimensional state:
         * 2n+1 = 19 sigma points.
         */

        sigmaPoints[0] =
            x;

        const float scale =
            0.25f;

        for (int k = 0; k < 9; k++)
        {
            sigmaPoints[k+1] =
                x;

            sigmaPoints[k+10] =
                x;

            float spread =
                sqrtf(
                    fabsf(P(k,k))
                ) * scale;

            sigmaPoints[k+1](k)
                += spread;

            sigmaPoints[k+10](k)
                -= spread;
        }
    }
};

// ============================================================
// UPF ESTIMATOR
// ============================================================

class UPFEstimator
{
public:

    UKFParticle particles[
        NUM_UPF_PARTICLES
    ];

    float weights[
        NUM_UPF_PARTICLES
    ];

    int particleCount;

    float sigmaUwb;

    UPFEstimator()
    {
        reset();
    }

    void reset()
    {
        particleCount =
            NUM_UPF_PARTICLES;

        sigmaUwb =
            0.1f;

        float initialWeight =
            1.0f /
            float(NUM_UPF_PARTICLES);

        for (int i = 0;
             i < NUM_UPF_PARTICLES;
             i++)
        {
            particles[i].reset();

            particles[i].weight =
                initialWeight;

            weights[i] =
                initialWeight;
        }
    }

    void initialize(
        float initialRange)
    {
        reset();

        /*
         * 4 azimuth
         * 2 altitude
         * 4 heading
         *
         * 32 hypotheses.
         */

        int index = 0;

        for (int az = 0; az < 4; az++)
        {
            for (int el = 0; el < 2; el++)
            {
                for (int h = 0; h < 4; h++)
                {
                    if (index >=
                        NUM_UPF_PARTICLES)
                        break;

                    float azimuth =
                        -PI +
                        az *
                        (2.0f * PI / 4.0f);

                    float altitude =
                        -PI/4.0f +
                        el *
                        (PI/2.0f);

                    float heading =
                        -PI +
                        h *
                        (2.0f * PI / 4.0f);

                    particles[index].x(0) =
                        initialRange;

                    particles[index].x(1) =
                        azimuth;

                    particles[index].x(2) =
                        altitude;

                    particles[index].x(3) =
                        heading;

                    particles[index].weight =
                        1.0f /
                        float(NUM_UPF_PARTICLES);

                    index++;
                }
            }
        }
    }

    void update(
        const Vec4 &dx,
        float range)
    {
        float total =
            0.0f;

        for (int i = 0;
             i < particleCount;
             i++)
        {
            particles[i].predict(dx);

            particles[i].update(
                range
            );

            weights[i] =
                particles[i].weight;

            total +=
                weights[i];
        }

        if (total < 1e-30f)
            total = 1e-30f;

        /*
         * Normalize particle weights.
         */

        for (int i = 0;
             i < particleCount;
             i++)
        {
            weights[i] /=
                total;

            particles[i].weight =
                weights[i];
        }

        /*
         * Branch-kill style workload.
         *
         * We retain the best hypothesis
         * and copy it into very-low-weight
         * branches.
         */

        int best =
            0;

        for (int i = 1;
             i < particleCount;
             i++)
        {
            if (weights[i] >
                weights[best])
            {
                best = i;
            }
        }

        for (int i = 0;
             i < particleCount;
             i++)
        {
            if (weights[i] < 0.01f)
            {
                particles[i] =
                    particles[best];

                weights[i] =
                    weights[best];
            }
        }
    }

    Vec4 getBestEstimate()
    {
        int best = 0;

        for (int i = 1;
             i < particleCount;
             i++)
        {
            if (weights[i] >
                weights[best])
            {
                best = i;
            }
        }

        Vec4 result{};

        for (int i = 0; i < 4; i++)
            result(i) =
                particles[best].x(i);

        return result;
    }
};

// ============================================================
// FIVE INDEPENDENT ALGEBRAIC ESTIMATORS
// ============================================================

AlgebraicEstimator algebraic[
    NUM_ALG_ESTIMATORS
];

// ONE UPF

UPFEstimator upf;

// ============================================================
// SYSTEM WORKLOAD
// ============================================================

volatile float imuAccumulator = 0.0f;

void simulateIMUWorkload()
{
#if ENABLE_IMU_TEST

    /*
     * Represents basic IMU processing:
     *
     * accelerometer
     * gyroscope
     * orientation update
     *
     * This is intentionally lightweight.
     *
     * Replace this with the actual IMU
     * driver/filter later.
     */

    float ax = 0.12f;
    float ay = 0.05f;
    float az = 9.81f;

    float gx = 0.01f;
    float gy = 0.02f;
    float gz = 0.03f;

    float magnitude =
        sqrtf(
            ax*ax +
            ay*ay +
            az*az
        );

    float gyroMagnitude =
        sqrtf(
            gx*gx +
            gy*gy +
            gz*gz
        );

    imuAccumulator +=
        magnitude +
        gyroMagnitude;

#endif
}

// ============================================================
// UART TELEMETRY
// ============================================================

void sendTelemetry()
{
#if ENABLE_UART_TEST

    Vec4 estimate =
        upf.getBestEstimate();

    Serial.printf(
        "UPF,"
        "%.4f,"
        "%.4f,"
        "%.4f,"
        "%.4f\n",
        estimate(0),
        estimate(1),
        estimate(2),
        estimate(3)
    );

#endif
}

// ============================================================
// WIFI WORKLOAD
// ============================================================

void performWiFiWorkload()
{
#if ENABLE_WIFI_TEST

    /*
     * Do NOT reconnect every estimator cycle.
     *
     * This represents the CPU activity of checking
     * the communication system.
     */

    if (WiFi.status() == WL_CONNECTED)
    {
        volatile int rssi =
            WiFi.RSSI();

        (void)rssi;
    }

#endif
}

// ============================================================
// BENCHMARK RESULTS
// ============================================================

struct BenchmarkResult
{
    float averageTimeUs;

    uint32_t worstTimeUs;

    uint32_t deadlineMisses;

    uint32_t freeHeap;

    uint32_t largestBlock;

    uint32_t minimumHeap;
};

BenchmarkResult results;

// ============================================================
// ESTIMATOR TASK
// ============================================================

void estimatorTask(
    void *parameter)
{
    estimatorTaskRunning =
        true;

    /*
     * Initial conditions.
     */

    upf.initialize(
        3.5f
    );

    for (int i = 0;
         i < NUM_ALG_ESTIMATORS;
         i++)
    {
        algebraic[i].reset(
            3.5f
        );
    }

    /*
     * Timing.
     */

    uint64_t totalUs =
        0;

    uint32_t worstUs =
        0;

    uint32_t misses =
        0;

    uint32_t minimumHeap =
        UINT32_MAX;

    /*
     * Run the benchmark at several
     * target rates.
     */

    const int rates[] =
    {
        10,
        20,
        30,
        50,
        100
    };

    for (int rateIndex = 0;
         rateIndex < 5;
         rateIndex++)
    {
        int frequency =
            rates[rateIndex];

        uint32_t periodUs =
            1000000UL /
            frequency;

        totalUs = 0;
        worstUs = 0;
        misses = 0;

        Serial.println();
        Serial.println(
            "======================================"
        );

        Serial.printf(
            "TARGET RATE: %d Hz\n",
            frequency
        );

        Serial.println(
            "======================================"
        );

        for (int step = 0;
             step < NUM_STEPS;
             step++)
        {
            uint32_t cycleStart =
                micros();

            /*
             * Synthetic odometry.
             *
             * Same input is intentionally sent
             * to every ALG estimator so that
             * each estimator is independent.
             */

            Vec4 dxHost{};
            Vec4 dxConnected{};

            dxHost(0) =
                0.01f;

            dxHost(1) =
                0.002f;

            dxHost(2) =
                0.001f;

            dxHost(3) =
                0.001f;

            dxConnected(0) =
                0.008f;

            dxConnected(1) =
                0.003f;

            dxConnected(2) =
                0.002f;

            dxConnected(3) =
                0.002f;

            float uwbRange =
                3.5f +
                0.01f *
                sinf(
                    step * 0.1f
                );

            /*
             * ------------------------------------------------
             * RUN ALL FIVE ALGEBRAIC ESTIMATORS
             * ------------------------------------------------
             */

            for (int i = 0;
                 i < NUM_ALG_ESTIMATORS;
                 i++)
            {
                algebraic[i].update(
                    uwbRange,
                    dxHost,
                    dxConnected
                );
            }

            /*
             * ------------------------------------------------
             * RUN ONE UPF
             * ------------------------------------------------
             */

            upf.update(
                dxConnected,
                uwbRange
            );

            /*
             * ------------------------------------------------
             * SYSTEM WORKLOAD
             * ------------------------------------------------
             */

            simulateIMUWorkload();

            performWiFiWorkload();

            /*
             * Telemetry only every 10 cycles
             * to prevent Serial from dominating
             * the benchmark.
             */

            if (step % 10 == 0)
                sendTelemetry();

            /*
             * END OF CYCLE
             */

            uint32_t elapsed =
                micros() -
                cycleStart;

            totalUs +=
                elapsed;

            if (elapsed >
                worstUs)
            {
                worstUs =
                    elapsed;
            }

            if (elapsed >
                periodUs)
            {
                misses++;
            }

            uint32_t heap =
                ESP.getFreeHeap();

            if (heap <
                minimumHeap)
            {
                minimumHeap =
                    heap;
            }

            /*
             * Maintain requested rate.
             */

            if (elapsed <
                periodUs)
            {
                delayMicroseconds(
                    periodUs -
                    elapsed
                );
            }
        }

        float average =
            float(totalUs) /
            float(NUM_STEPS);

        float utilization =
            (average /
             float(periodUs))
            * 100.0f;

        Serial.printf(
            "\n%d Hz RESULTS\n",
            frequency
        );

        Serial.printf(
            "Average cycle: %.2f us\n",
            average
        );

        Serial.printf(
            "Average cycle: %.3f ms\n",
            average / 1000.0f
        );

        Serial.printf(
            "Worst cycle: %lu us\n",
            (unsigned long)worstUs
        );

        Serial.printf(
            "Budget: %lu us\n",
            (unsigned long)periodUs
        );

        Serial.printf(
            "Estimated utilization: %.2f %%\n",
            utilization
        );

        Serial.printf(
            "Deadline misses: %lu / %d\n",
            (unsigned long)misses,
            NUM_STEPS
        );

        Serial.printf(
            "Minimum free heap: %lu bytes\n",
            (unsigned long)minimumHeap
        );

        Serial.printf(
            "Largest free block: %lu bytes\n",
            (unsigned long)
            heap_caps_get_largest_free_block(
                MALLOC_CAP_8BIT
            )
        );

        /*
         * A short pause between benchmark rates.
         */

        delay(1000);
    }

    estimatorTaskRunning =
        false;

    Serial.println();
    Serial.println(
        "======================================"
    );

    Serial.println(
        "BENCHMARK COMPLETE"
    );

    Serial.println(
        "======================================"
    );

    vTaskDelete(NULL);
}

// ============================================================
// SYSTEM TASK
// ============================================================

void systemTask(
    void *parameter)
{
    systemTaskRunning =
        true;

    while (true)
    {
        /*
         * System-side workload.
         *
         * In the final hardware test this task
         * should contain:
         *
         *   IMU driver
         *   WiFi
         *   UART
         *   external odometry
         */

        simulateIMUWorkload();

        performWiFiWorkload();

        systemSteps++;

        delay(1);
    }
}

// ============================================================
// SETUP
// ============================================================

void setup()
{
    Serial.begin(
        TELEMETRY_BAUD
    );

    delay(1500);

    Serial.println();
    Serial.println(
        "======================================"
    );

    Serial.println(
        "ESP32-S3 UPF-RPE RESOURCE BENCHMARK"
    );

    Serial.println(
        "======================================"
    );

    Serial.printf(
        "CPU frequency: %u MHz\n",
        ESP.getCpuFreqMHz()
    );

    Serial.printf(
        "Chip revision: %d\n",
        ESP.getChipRevision()
    );

    Serial.printf(
        "Free heap: %u bytes\n",
        ESP.getFreeHeap()
    );

    Serial.printf(
        "Largest free block: %u bytes\n",
        heap_caps_get_largest_free_block(
            MALLOC_CAP_8BIT
        )
    );

    Serial.printf(
        "NUM ALG: %d\n",
        NUM_ALG_ESTIMATORS
    );

    Serial.printf(
        "NUM UPF particles: %d\n",
        NUM_UPF_PARTICLES
    );

    Serial.println();

#if ENABLE_WIFI_TEST

    Serial.println(
        "Starting WiFi subsystem..."
    );

    /*
     * For the benchmark we initialize WiFi
     * but do not require a real network.
     */

    WiFi.mode(
        WIFI_STA
    );

#endif

    /*
     * Static object memory.
     */

    Serial.println(
        "--------------------------------------"
    );

    Serial.printf(
        "sizeof(AlgebraicEstimator): %u bytes\n",
        sizeof(AlgebraicEstimator)
    );

    Serial.printf(
        "sizeof(UKFParticle): %u bytes\n",
        sizeof(UKFParticle)
    );

    Serial.printf(
        "sizeof(UPFEstimator): %u bytes\n",
        sizeof(UPFEstimator)
    );

    Serial.printf(
        "5 ALG total: %u bytes\n",
        sizeof(algebraic)
    );

    Serial.printf(
        "UPF total: %u bytes\n",
        sizeof(upf)
    );

    Serial.printf(
        "Estimator static total: %u bytes\n",
        sizeof(algebraic)
        +
        sizeof(upf)
    );

    Serial.println(
        "--------------------------------------"
    );

    /*
     * Create system task on Core 0.
     */

    xTaskCreatePinnedToCore(
        systemTask,
        "SYSTEM",
        8192,
        NULL,
        1,
        &systemTaskHandle,
        SYSTEM_CORE
    );

    /*
     * Create estimator task on Core 1.
     */

    xTaskCreatePinnedToCore(
        estimatorTask,
        "ESTIMATOR",
        16384,
        NULL,
        2,
        &estimatorTaskHandle,
        ESTIMATOR_CORE
    );
}

// ============================================================
// LOOP
// ============================================================

void loop()
{
    /*
     * Everything important is running
     * inside FreeRTOS tasks.
     */

    delay(1000);
}
