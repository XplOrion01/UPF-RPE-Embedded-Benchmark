/*
 * ============================================================
 * TEENSY 4.1 UPF-RPE RESOURCE BENCHMARK
 * ============================================================
 *
 * Target:
 *
 *   Teensy 4.1
 *   NXP i.MX RT1062
 *   ARM Cortex-M7
 *   600 MHz
 *
 * Workload:
 *
 *   5 x Algebraic 4DoF estimators
 *   1 x UPF estimator
 *   32 x UPF particles
 *   IMU workload
 *   communication workload
 *   UART telemetry
 *
 * IMPORTANT:
 *
 * This is a RESOURCE / TIMING benchmark.
 *
 * It is NOT yet the final line-by-line C++ port
 * of Yuri's Python UPF-RPE implementation.
 *
 * ============================================================
 */

#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>


// ============================================================
// CONFIGURATION
// ============================================================

#define NUM_ALG_ESTIMATORS 5
#define NUM_UPF_PARTICLES  32

#define NUM_STEPS 200

#define ENABLE_WIFI_TEST   1
#define ENABLE_IMU_TEST    1
#define ENABLE_UART_TEST   1

#define TELEMETRY_BAUD     115200


// ============================================================
// BASIC TYPES
// ============================================================

struct Vec4
{
    float v[4];

    float &operator()(int i)
    {
        return v[i];
    }

    const float &operator()(int i) const
    {
        return v[i];
    }
};


struct Mat4
{
    float m[4][4];

    float &operator()(int r, int c)
    {
        return m[r][c];
    }

    const float &operator()(int r, int c) const
    {
        return m[r][c];
    }
};


struct Vec9
{
    float v[9];

    float &operator()(int i)
    {
        return v[i];
    }

    const float &operator()(int i) const
    {
        return v[i];
    }
};


struct Mat9
{
    float m[9][9];

    float &operator()(int r, int c)
    {
        return m[r][c];
    }

    const float &operator()(int r, int c) const
    {
        return m[r][c];
    }
};


// ============================================================
// GLOBALS
// ============================================================

volatile uint32_t estimatorSteps = 0;
volatile uint32_t systemSteps = 0;

volatile float imuAccumulator = 0.0f;

volatile uint32_t deadlineMisses = 0;


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
            get4DRotation(
                hostPrevious(3)
            );


        Mat4 Cc =
            get4DRotation(
                connectedPrevious(3)
            );


        Vec4 hostCurrent =
            hostPrevious;

        Vec4 connectedCurrent =
            connectedPrevious;


        Vec4 dh =
            matVec4(
                Ch,
                dxHost
            );


        Vec4 dc =
            matVec4(
                Cc,
                dxConnected
            );


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
            hostOdom[count] =
                hostCurrent;

            connectedOdom[count] =
                connectedCurrent;

            eps[count] =
                e;

            distance[count] =
                d;

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
         * Replace this with the complete translation of
         * Yuri's AlgebraicMethod4DoF.find_relative_pose()
         * in the final implementation.
         */

        float sx = 0.0f;
        float sy = 0.0f;
        float sz = 0.0f;


        for (int i = 1; i < HORIZON; i++)
        {
            sx +=
                connectedOdom[i](0)
                -
                hostOdom[i](0);

            sy +=
                connectedOdom[i](1)
                -
                hostOdom[i](1);

            sz +=
                connectedOdom[i](2)
                -
                hostOdom[i](2);
        }


        float inv =
            1.0f /
            float(HORIZON - 1);


        x_ca_0_alg(0) =
            sx * inv;

        x_ca_0_alg(1) =
            sy * inv;

        x_ca_0_alg(2) =
            sz * inv;


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
            matVec4(
                Ch,
                difference
            );


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
            {
                P(i,j) =
                    (i == j)
                    ? 0.1f
                    : 0.0f;
            }
        }

        weight = 1.0f;

        sigmaUwb = 0.1f;
    }


    void predict(
        const Vec4 &dx)
    {
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
            measuredRange -
            predicted;


        float variance =
            sigmaUwb*sigmaUwb +
            0.01f;


        float likelihood =
            expf(
                -0.5f *
                error*error /
                variance
            );


        if (likelihood < 1e-20f)
            likelihood = 1e-20f;


        weight *= likelihood;


        float K =
            P(0,0) /
            (
                P(0,0) +
                variance
            );


        x(0) +=
            K * error;


        P(0,0) =
            (1.0f - K) *
            P(0,0);
    }


    void generateSigmaPoints()
    {
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
                )
                *
                scale;


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


        for (
            int i = 0;
            i < NUM_UPF_PARTICLES;
            i++
        )
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


        int index = 0;


        for (int az = 0; az < 4; az++)
        {
            for (int el = 0; el < 2; el++)
            {
                for (int h = 0; h < 4; h++)
                {
                    if (
                        index >=
                        NUM_UPF_PARTICLES
                    )
                        break;


                    float azimuth =
                        -PI +
                        az *
                        (
                            2.0f *
                            PI /
                            4.0f
                        );


                    float altitude =
                        -PI/4.0f +
                        el *
                        (
                            PI/2.0f
                        );


                    float heading =
                        -PI +
                        h *
                        (
                            2.0f *
                            PI /
                            4.0f
                        );


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
                        float(
                            NUM_UPF_PARTICLES
                        );


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


        for (
            int i = 0;
            i < particleCount;
            i++
        )
        {
            particles[i].predict(
                dx
            );


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


        for (
            int i = 0;
            i < particleCount;
            i++
        )
        {
            weights[i] /=
                total;

            particles[i].weight =
                weights[i];
        }


        int best =
            0;


        for (
            int i = 1;
            i < particleCount;
            i++
        )
        {
            if (
                weights[i] >
                weights[best]
            )
            {
                best = i;
            }
        }


        for (
            int i = 0;
            i < particleCount;
            i++
        )
        {
            if (
                weights[i] < 0.01f
            )
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
        int best =
            0;


        for (
            int i = 1;
            i < particleCount;
            i++
        )
        {
            if (
                weights[i] >
                weights[best]
            )
            {
                best = i;
            }
        }


        Vec4 result{};


        for (int i = 0; i < 4; i++)
        {
            result(i) =
                particles[best].x(i);
        }


        return result;
    }
};


// ============================================================
// GLOBAL ESTIMATORS
// ============================================================

AlgebraicEstimator algebraic[
    NUM_ALG_ESTIMATORS
];


UPFEstimator upf;


// ============================================================
// IMU WORKLOAD
// ============================================================

void simulateIMUWorkload()
{
#if ENABLE_IMU_TEST

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
// COMMUNICATION WORKLOAD
// ============================================================
//
// Teensy 4.1 has no onboard WiFi.
//
// This represents lightweight communication CPU activity.
//
// Replace later with the actual Ethernet/WiFi/UART/CAN
// communication workload.
//
// ============================================================

volatile uint32_t communicationCounter = 0;


void performWiFiWorkload()
{
#if ENABLE_WIFI_TEST

    communicationCounter++;

    if (
        communicationCounter >
        1000000UL
    )
    {
        communicationCounter = 0;
    }

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
// HEAP MEASUREMENT
// ============================================================
//
// Teensy does not provide ESP32's:
//
//   ESP.getFreeHeap()
//   heap_caps_get_largest_free_block()
//
// Therefore we deliberately do NOT claim an equivalent
// measurement here.
//
// Static estimator memory is reported exactly with sizeof().
//
// ============================================================

extern "C" char* sbrk(int incr);


uint32_t getCurrentHeapEnd()
{
    return (uint32_t)(uintptr_t)sbrk(0);
}


// ============================================================
// RUN ONE ESTIMATION CYCLE
// ============================================================

static inline void runEstimatorCycle(
    int step)
{
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
     * FIVE ALGEBRAIC ESTIMATORS
     */

    for (
        int i = 0;
        i < NUM_ALG_ESTIMATORS;
        i++
    )
    {
        algebraic[i].update(
            uwbRange,
            dxHost,
            dxConnected
        );
    }


    /*
     * ONE UPF
     */

    upf.update(
        dxConnected,
        uwbRange
    );


    /*
     * SYSTEM WORKLOAD
     */

    simulateIMUWorkload();

    performWiFiWorkload();


    /*
     * UART telemetry.
     */

    if (step % 10 == 0)
        sendTelemetry();


    estimatorSteps++;
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
        "TEENSY 4.1 UPF-RPE RESOURCE BENCHMARK"
    );

    Serial.println(
        "======================================"
    );


    Serial.printf(
        "CPU frequency: %lu MHz\n",
        (unsigned long)(
            F_CPU / 1000000UL
        )
    );


    Serial.println(
        "CPU architecture: ARM Cortex-M7"
    );


    Serial.println(
        "CPU cores: 1"
    );


    Serial.println(
        "WiFi: NOT ONBOARD"
    );


    Serial.println(
        "Communication workload: SIMULATED"
    );


    Serial.println();


    /*
     * Initialize estimators.
     */

    upf.initialize(
        3.5f
    );


    for (
        int i = 0;
        i < NUM_ALG_ESTIMATORS;
        i++
    )
    {
        algebraic[i].reset(
            3.5f
        );
    }


    /*
     * Static memory information.
     */

    Serial.println(
        "--------------------------------------"
    );


    Serial.printf(
        "sizeof(AlgebraicEstimator): %u bytes\n",
        (unsigned)sizeof(
            AlgebraicEstimator
        )
    );


    Serial.printf(
        "sizeof(UKFParticle): %u bytes\n",
        (unsigned)sizeof(
            UKFParticle
        )
    );


    Serial.printf(
        "sizeof(UPFEstimator): %u bytes\n",
        (unsigned)sizeof(
            UPFEstimator
        )
    );


    Serial.printf(
        "5 ALG total: %u bytes\n",
        (unsigned)sizeof(
            algebraic
        )
    );


    Serial.printf(
        "UPF total: %u bytes\n",
        (unsigned)sizeof(
            upf
        )
    );


    Serial.printf(
        "Estimator static total: %u bytes\n",
        (unsigned)(
            sizeof(algebraic)
            +
            sizeof(upf)
        )
    );


    Serial.println(
        "--------------------------------------"
    );


    /*
     * Initial heap break.
     */

    Serial.printf(
        "Initial heap break: 0x%08lX\n",
        (unsigned long)
        getCurrentHeapEnd()
    );


    Serial.println();

    Serial.println(
        "Starting benchmark..."
    );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
    /*
     * Run benchmark at several target rates.
     */

    const int rates[] =
    {
        10,
        20,
        30,
        50,
        100
    };


    for (
        int rateIndex = 0;
        rateIndex < 5;
        rateIndex++
    )
    {
        int frequency =
            rates[rateIndex];


        uint32_t periodUs =
            1000000UL /
            frequency;


        uint64_t totalUs =
            0;


        uint32_t worstUs =
            0;


        uint32_t misses =
            0;


        uint32_t heapBefore =
            getCurrentHeapEnd();


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


        /*
         * Reset estimator state for each
         * benchmark rate.
         */

        upf.initialize(
            3.5f
        );


        for (
            int i = 0;
            i < NUM_ALG_ESTIMATORS;
            i++
        )
        {
            algebraic[i].reset(
                3.5f
            );
        }


        for (
            int step = 0;
            step < NUM_STEPS;
            step++
        )
        {
            uint32_t cycleStart =
                micros();


            runEstimatorCycle(
                step
            );


            uint32_t elapsed =
                micros() -
                cycleStart;


            totalUs +=
                elapsed;


            if (
                elapsed >
                worstUs
            )
            {
                worstUs =
                    elapsed;
            }


            if (
                elapsed >
                periodUs
            )
            {
                misses++;
            }


            /*
             * Maintain requested rate.
             */

            if (
                elapsed <
                periodUs
            )
            {
                delayMicroseconds(
                    periodUs -
                    elapsed
                );
            }
        }


        uint32_t heapAfter =
            getCurrentHeapEnd();


        float average =
            float(totalUs) /
            float(NUM_STEPS);


        float utilization =
            (
                average /
                float(periodUs)
            )
            *
            100.0f;


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
            (unsigned long)
            worstUs
        );


        Serial.printf(
            "Budget: %lu us\n",
            (unsigned long)
            periodUs
        );


        Serial.printf(
            "Estimated utilization: %.2f %%\n",
            utilization
        );


        Serial.printf(
            "Deadline misses: %lu / %d\n",
            (unsigned long)
            misses,
            NUM_STEPS
        );


        Serial.printf(
            "Heap break before: 0x%08lX\n",
            (unsigned long)
            heapBefore
        );


        Serial.printf(
            "Heap break after:  0x%08lX\n",
            (unsigned long)
            heapAfter
        );


        Serial.println();


        /*
         * Pause before next rate.
         */

        delay(1000);
    }


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


    /*
     * Stop loop from repeating benchmark.
     */

    while (true)
    {
        delay(1000);
    }
}
