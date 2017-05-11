#include "Mandel.h"
#include <cuda_runtime.h>

// Checks if c lies within the main cardioid or period 2 component
#define QUICK_CHECKS 1

// Manually unroll the iteration loops
// Possible values are: 1 (disable), 4, 8, 12, 16, 20, or 32
#define UNROLL 16

// For block-scheduler
static __device__ unsigned int d_blockCounter = 0;

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static __device__ __constant__ color4 kColorMap[256] = {
#include "Maps/damien3.map"
};

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
static int divup(int x, int n)
{
    return (x + (n - 1)) / n;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
#if QUICK_CHECKS
template <typename Float>
inline __device__ int quick_checks(Float x, Float y)
{
    // main cardioid
    if ((x * x + y * y) * ((Float)8.0 * (x * x + y * y) - (Float)3.0) + x < (Float)3.0/(Float)32.0)
        return 1;

    // period 2 component
    if ((x + (Float)1.0) * (x + (Float)1.0) + y * y < (Float)1.0/(Float)16.0)
        return 1;

    return 0;
}
#endif

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template<typename T, typename Float>
inline __host__ __device__ T lerp(T a, T b, Float t)
{
    return a * ((Float)1.0 - t) + b * t;
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
inline __host__ __device__ color4 lerp_color(color4 c1, color4 c2, Float t)
{
    return make_color4(
        (unsigned char)round( lerp((Float)c1.b, (Float)c2.b, t) ),
        (unsigned char)round( lerp((Float)c1.g, (Float)c2.g, t) ),
        (unsigned char)round( lerp((Float)c1.r, (Float)c2.r, t) ),
        (unsigned char)round( lerp((Float)c1.a, (Float)c2.a, t) )
        );
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __device__ unsigned int levelset(
    const Float         cx, 
    const Float         cy, 
    const unsigned int  maxit,
    const Float         bailout,
    Float*              mag)
{
    Float x = 0;
    Float y = 0;
    Float x2 = 0;
    Float y2 = 0;

#if QUICK_CHECKS
    if (quick_checks(cx, cy))
    {
        return maxit;
    }
#endif

    unsigned int i = 0;
    unsigned int k = 0;

#define Iterate()                       \
    {	y  = (Float)2.0 * x * y + cy;   \
        x  = x2 - y2 + cx;              \
        y2 = y*y;                       \
        x2 = x*x;                       \
        k++;                            \
        if (x2 + y2 > bailout)          \
            break;                      \
    }

    for (i = 0, k = 0; i < maxit; i += UNROLL, k = 0)
    {
        Iterate(); // 1
#if (UNROLL > 1)
        Iterate(); // 2
        Iterate(); // 3
        Iterate(); // 4
#if (UNROLL > 4)
        Iterate(); // 5
        Iterate(); // 6
        Iterate(); // 7
        Iterate(); // 8
#if (UNROLL > 8)
        Iterate(); // 9
        Iterate(); // 10
        Iterate(); // 11
        Iterate(); // 12
#if (UNROLL > 12)
        Iterate(); // 13
        Iterate(); // 14
        Iterate(); // 15
        Iterate(); // 16
#if (UNROLL > 16)
        Iterate(); // 17
        Iterate(); // 18
        Iterate(); // 19
        Iterate(); // 20
#if (UNROLL > 20)
        Iterate(); // 21
        Iterate(); // 22
        Iterate(); // 23
        Iterate(); // 24
        Iterate(); // 25
        Iterate(); // 26
        Iterate(); // 27
        Iterate(); // 28
        Iterate(); // 29
        Iterate(); // 30
        Iterate(); // 31
        Iterate(); // 32
#endif
#endif
#endif
#endif
#endif
#endif
    }

#undef Iterate

    if (mag)
        mag[0] = x2 + y2;

    return min(i + k, maxit);
}

//--------------------------------------------------------------------------------------------------
// Estimate the distance of (cx,cy) to the mandelbrot set
// NOTE: the larger the value of maxit, the better the estimate
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __device__ unsigned int distest(
    const Float         cx,
    const Float         cy,
    const unsigned int  maxit,
    Float*              dist)
{
    Float x = 0;
    Float y = 0;
    Float dx = 1;
    Float dy = 0;
    Float x2 = 0;
    Float y2 = 0;

#if QUICK_CHECKS
    if (quick_checks(cx, cy))
    {
        return maxit;
    }
#endif

    unsigned int i = 0;
    unsigned int k = 0;

#define Iterate()                                                       \
    {   const Float dt = (Float)2.0 * (x * dx - y * dy) + (Float)1.0;   \
        dy = (Float)2.0 * (y * dx + x * dy);                            \
        dx = dt;                                                        \
        y  = (Float)2.0 * x * y + cy;                                   \
        x  = x2 - y2 + cx;                                              \
        y2 = y*y;                                                       \
        x2 = x*x;                                                       \
        k++;                                                            \
        if (x2 + y2 > (Float)4.0)                                       \
            break;                                                      \
    }

    for (i = 0, k = 0; i < maxit; i += UNROLL, k = 0)
    {
        Iterate(); // 1
#if (UNROLL > 1)
        Iterate(); // 2
        Iterate(); // 3
        Iterate(); // 4
#if (UNROLL > 4)
        Iterate(); // 5
        Iterate(); // 6
        Iterate(); // 7
        Iterate(); // 8
#if (UNROLL > 8)
        Iterate(); // 9
        Iterate(); // 10
        Iterate(); // 11
        Iterate(); // 12
#if (UNROLL > 12)
        Iterate(); // 13
        Iterate(); // 14
        Iterate(); // 15
        Iterate(); // 16
#if (UNROLL > 16)
        Iterate(); // 17
        Iterate(); // 18
        Iterate(); // 19
        Iterate(); // 20
#if (UNROLL > 20)
        Iterate(); // 21
        Iterate(); // 22
        Iterate(); // 23
        Iterate(); // 24
        Iterate(); // 25
        Iterate(); // 26
        Iterate(); // 27
        Iterate(); // 28
        Iterate(); // 29
        Iterate(); // 30
        Iterate(); // 31
        Iterate(); // 32
#endif
#endif
#endif
#endif
#endif
#endif
    }

#undef Iterate

    //
    // The distance d(c) of a point in the parameter plane to the mandelbrot set M is given by
    //
    //    d(c) = lim_{n->inf} ( 2 log(|z_n|) |z_n| / |dz_n| )
    //
    // TODO:
    // iterate a few more times to increase the accuracy of the estimate?!?!
    //
    const Float z2 = x2 + y2;
    const Float dz2 = dx * dx + dy * dy;

    dist[0] = log(z2) * sqrt(z2 / dz2);

    return min(i + k, maxit);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __device__ void kernel_levelset_per_thread(
    color4*             image,
    const unsigned int  imageW,
    const unsigned int  imageH,
    const Float         minX,
    const Float         maxY,
    const Float         delta,
    const unsigned int  maxit,
    const unsigned int  blockX,
    const unsigned int  blockY)
{
    const unsigned int ix = threadIdx.x + blockX * blockDim.x;
    const unsigned int iy = threadIdx.y + blockY * blockDim.y;

    if (ix < imageW && iy < imageH)
    {
        const unsigned int c = levelset(minX + delta * ix, maxY - delta * iy, maxit, (Float)4.0, (Float*)nullptr);

        if (c < maxit)
        {
            image[ix + iy * imageW] = kColorMap[c & 0xff];
        }
        else
        {
            image[ix + iy * imageW] = make_color4(0,0,0);
        }
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __global__ void kernel_levelset(
    color4*             image,      // device image
    const unsigned int  imageW,     // image dimensions
    const unsigned int  imageH,
    const Float         minX,       // plane position
    const Float         maxY,
    const Float         delta,      // pixel dimensions in plane
    const unsigned int  maxit)      // max nr of iterations
{
    kernel_levelset_per_thread(
        image, imageW, imageH, minX, maxY, delta, maxit, blockIdx.x, blockIdx.y );
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __global__ void kernel_levelset_block_scheduling(
    color4*             image,      // device image
    const unsigned int  imageW,     // image dimensions
    const unsigned int  imageH,
    const Float         minX,       // plane position
    const Float         maxY,
    const Float         delta,      // pixel dimensions in plane
    const unsigned int  maxit,      // max nr of iterations
    const unsigned int  gridWidth,
    const unsigned int  numBlocks)
{
    __shared__ unsigned int blockIndex;
    __shared__ unsigned int blockX;
    __shared__ unsigned int blockY;

    for (;;)
    {
        // Thread 0,0 computes the grid position for this thread block
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            blockIndex = atomicAdd(&d_blockCounter, 1);
            blockX = blockIndex % gridWidth;
            blockY = blockIndex / gridWidth;
        }

        // Then wait until all threads in this block can read their new grid position
        __syncthreads();

        // Exit the loop if all blocks have been processed
        if (blockIndex >= numBlocks)
        {
            break;
        }

        // Eventually (all) the threads in this block do some work
        kernel_levelset_per_thread(
            image, imageW, imageH, minX, maxY, delta, maxit, blockX, blockY );
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __device__ void kernel_levelset_cont_per_thread(
    color4*             image,      // device image
    const unsigned int  imageW,     // image dimensions
    const unsigned int  imageH,
    const Float         minX,       // plane position
    const Float         maxY,
    const Float         delta,      // pixel dimensions in plane
    const unsigned int  maxit,      // max nr of iterations
    const unsigned int  blockX,
    const unsigned int  blockY)
{
    const unsigned int ix = threadIdx.x + blockX * blockDim.x;
    const unsigned int iy = threadIdx.y + blockY * blockDim.y;

    if (ix < imageW && iy < imageH)
    {
        const unsigned int index = ix + iy * imageW;

        Float mag = 0;
//      const unsigned int c = levelset(minX + delta * ix, maxY - delta * iy, maxit, (Float)256.0, &mag);
        const unsigned int c = levelset(minX + delta * ix, maxY - delta * iy, maxit, (Float)4.0, &mag);

        if (c < maxit)
        {
            //
            // See:
            // http://en.wikipedia.org/wiki/Mandelbrot_set#Continuous_.28smooth.29_coloring
            //
#if 0
            const Float nu = c - log2((Float)0.5 * log2(mag));
            const Float k = ceilf(nu);
            const Float f = k - nu;

            const int i = (int)k;

            image[index] = lerp_color(kColorMap[(i + 1) & 0xff], kColorMap[i & 0xff], f);
#else
            const Float v = (Float)0.025 * (c - log2((Float)0.5 * log2(mag)));

            image[index].b = roundf((Float)255.0 * ( cos(sqrt((Float)7.0) * v + (Float)2.0) * (Float)0.5 + (Float)0.5 ));
            image[index].g = roundf((Float)255.0 * ( cos(sqrt((Float)6.0) * v + (Float)1.0) * (Float)0.5 + (Float)0.5 ));
            image[index].r = roundf((Float)255.0 * ( cos(sqrt((Float)5.0) * v + (Float)0.0) * (Float)0.5 + (Float)0.5 ));
            image[index].a = 255;
#endif
        }
        else
        {
            image[index] = make_color4(0,0,0);
        }
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __global__ void kernel_levelset_cont(
    color4*             image,      // device image
    const unsigned int  imageW,     // image dimensions
    const unsigned int  imageH,
    const Float         minX,       // plane position
    const Float         maxY,
    const Float         delta,      // pixel dimensions in plane
    const unsigned int  maxit)      // max nr of iterations
{
    kernel_levelset_cont_per_thread(
        image, imageW, imageH, minX, maxY, delta, maxit, blockIdx.x, blockIdx.y );
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __global__ void kernel_levelset_cont_block_scheduling(
    color4*             image,      // device image
    const unsigned int  imageW,     // image dimensions
    const unsigned int  imageH,
    const Float         minX,       // plane position
    const Float         maxY,
    const Float         delta,      // pixel dimensions in plane
    const unsigned int  maxit,      // max nr of iterations
    const unsigned int  gridWidth,
    const unsigned int  numBlocks)
{
    __shared__ unsigned int blockIndex;
    __shared__ unsigned int blockX;
    __shared__ unsigned int blockY;

    for (;;)
    {
        // Thread 0,0 computes the grid position for this thread block
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            blockIndex = atomicAdd(&d_blockCounter, 1);
            blockX = blockIndex % gridWidth;
            blockY = blockIndex / gridWidth;
        }

        // Then wait until all threads in this block can read their new grid position
        __syncthreads();

        // Exit the loop if all blocks have been processed
        if (blockIndex >= numBlocks)
        {
            break;
        }

        // Eventually (all) the threads in this block do some work
        kernel_levelset_cont_per_thread(
            image, imageW, imageH, minX, maxY, delta, maxit, blockX, blockY );
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __device__ void kernel_distest_per_thread(
    color4*             image,      // device image
    const unsigned int  imageW,     // image dimensions
    const unsigned int  imageH,
    const Float         minX,       // plane position
    const Float         maxY,
    const Float         delta,      // pixel dimensions in plane
    const Float         scale,      // should be ~1/delta
    const unsigned int  maxit,      // max nr of iterations
    const unsigned int  blockX,
    const unsigned int  blockY)
{
    const unsigned int ix = threadIdx.x + blockX * blockDim.x;
    const unsigned int iy = threadIdx.y + blockY * blockDim.y;

    if (ix < imageW && iy < imageH)
    {
        Float dist;
        const unsigned int c = distest(minX + delta * ix, maxY - delta * iy, maxit, &dist);

        if (c == maxit)
        {
            image[ix + iy * imageW] = make_color4(0,0,0);
        }
        else
        {
            image[ix + iy * imageW]
                = make_color4(
                    255,
                    round(lerp((Float)0.0, (Float)192.0, saturate(pow(scale * dist, (Float)0.4)))),
                    0);
        }
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __global__ void kernel_distest(
    color4*             image,      // device image
    const unsigned int  imageW,     // image dimensions
    const unsigned int  imageH,
    const Float         minX,       // plane position
    const Float         maxY,
    const Float         delta,      // pixel dimensions in plane
    const Float         scale,      // should be ~1/delta
    const unsigned int  maxit)      // max nr of iterations
{
    kernel_distest_per_thread(
        image, imageW, imageH, minX, maxY, delta, scale, maxit, blockIdx.x, blockIdx.y);
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __global__ void kernel_distest_block_scheduling(
    color4*             image,      // device image
    const unsigned int  imageW,     // image dimensions
    const unsigned int  imageH,
    const Float         minX,       // plane position
    const Float         maxY,
    const Float         delta,      // pixel dimensions in plane
    const Float         scale,
    const unsigned int  maxit,      // max nr of iterations
    const unsigned int  gridWidth,
    const unsigned int  numBlocks)
{
    __shared__ unsigned int blockIndex;
    __shared__ unsigned int blockX;
    __shared__ unsigned int blockY;

    for (;;)
    {
        // Thread 0,0 computes the grid position for this thread block
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            blockIndex = atomicAdd(&d_blockCounter, 1);
            blockX = blockIndex % gridWidth;
            blockY = blockIndex / gridWidth;
        }

        // Then wait until all threads in this block can read their new grid position
        __syncthreads();

        // Exit the loop if all blocks have been processed
        if (blockIndex >= numBlocks)
        {
            break;
        }

        // Eventually (all) the threads in this block do some work
        kernel_distest_per_thread(
            image, imageW, imageH, minX, maxY, delta, scale, maxit, blockX, blockY );
    }
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static __global__ void kernel_cn(
    color4*             image,      // device image
    const unsigned int  imageW,     // image dimensions
    const unsigned int  imageH,
    const Float         minX,       // plane position
    const Float         maxY,
    const Float         delta,      // pixel dimensions in plane
    const unsigned int  maxit)      // max nr of iterations
{
    // Shared memory to hold the levelset values for a complete block
    extern __shared__ unsigned int smem[];

#define S(a,b) smem[(a) + blockDim.x * (b)]
#define u threadIdx.x
#define v threadIdx.y

    //
    // compute pixel position
    //
    const int ix = u - 1 + blockIdx.x * (blockDim.x - 2);
    const int iy = v - 1 + blockIdx.y * (blockDim.y - 2);

    //
    // compute levelset
    // store result in shared memory
    //
    if (/*-1 <= ix && */ ix <= imageW && /*-1 <= iy && */ iy <= imageH)
    {
        S(u,v) = levelset(minX + delta * ix, maxY - delta * iy, maxit, (Float)4.0, (Float*)nullptr);
    }

    //
    // synchronize
    // wait until all threads have computed the levelset for their position
    //
    __syncthreads();

    if ((/*-1 < ix &&*/ ix < imageW && /*-1 < iy &&*/ iy < imageH)
            && (0 < u && u + 1 < blockDim.x && 0 < v && v + 1 < blockDim.y))
    {
        const unsigned int index = ix + iy * imageW;

        const unsigned int c = S(u,v);

        if (c == maxit)
        {
            image[index] = make_color4(0,0,0);
        }
        else
        {
            unsigned int n = 0;

            if (S(u-1, v-1) < c) n++;
            if (S(u  , v-1) < c) n++;
            if (S(u+1, v-1) < c) n++;
            if (S(u-1, v  ) < c) n++;
            if (S(u+1, v  ) < c) n++;
            if (S(u-1, v+1) < c) n++;
            if (S(u  , v+1) < c) n++;
            if (S(u+1, v+1) < c) n++;

            image[index] = (n < 5) ? make_color4(0, 192, 0) : make_color4(255, 0, 0);
        }
    }

#undef v
#undef u
#undef S
}

//--------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------
template <typename Float>
static void mandel(
    color4*             image,
    const int           imageW,
    const int           imageH,
    const Float         cenX,
    const Float         cenY,
    const Float         radius,
    const unsigned int  maxit,
    const int           algorithm,
    const int           blockSceduling,
    const unsigned int  multiProcessorCount)
{
    //
    // Compute plane parameters used by the kernels
    //
    double delta = radius / imageW;
    double minX = cenX - delta * (Float)0.5 * imageW;
    double maxY = cenY + delta * (Float)0.5 * imageH;

    switch (algorithm)
    {
    case MANDEL_LEVELSET:
        if (blockSceduling)
        {
            //
            // zero block counter
            //
            unsigned int h_blockCounter = 0;
            cudaMemcpyToSymbol(
                d_blockCounter, &h_blockCounter, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

            dim3 dimBlock(16, 16);
            dim3 dimGrid(divup(imageW, dimBlock.x), divup(imageH, dimBlock.y));

            kernel_levelset_block_scheduling<<< multiProcessorCount, dimBlock >>>(
                image, imageW, imageH, (Float)minX, (Float)maxY, (Float)delta,
                maxit, dimGrid.x, dimGrid.x * dimGrid.y);
        }
        else
        {
            dim3 dimBlock(16, 16);
            dim3 dimGrid(divup(imageW, dimBlock.x), divup(imageH, dimBlock.y));

            kernel_levelset<<< dimGrid, dimBlock >>>
                (image, imageW, imageH, (Float)minX, (Float)maxY, (Float)delta, maxit);
        }
        break;

    case MANDEL_LEVELSETCONT:
        if (blockSceduling)
        {
            //
            // zero block counter
            //
            unsigned int h_blockCounter = 0;
            cudaMemcpyToSymbol(
                d_blockCounter, &h_blockCounter, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

            dim3 dimBlock(16, 16);
            dim3 dimGrid(divup(imageW, dimBlock.x), divup(imageH, dimBlock.y));

            kernel_levelset_cont_block_scheduling<<< multiProcessorCount, dimBlock >>>(
                image, imageW, imageH, (Float)minX, (Float)maxY, (Float)delta,
                maxit, dimGrid.x, dimGrid.x * dimGrid.y);
        }
        else
        {
            dim3 dimBlock(16, 16);
            dim3 dimGrid(divup(imageW, dimBlock.x), divup(imageH, dimBlock.y));

            kernel_levelset_cont<<< dimGrid, dimBlock >>>
                (image, imageW, imageH, (Float)minX, (Float)maxY, (Float)delta, maxit);
        }
        break;

    case MANDEL_DISTEST:
        if (blockSceduling)
        {
            //
            // zero block counter
            //
            unsigned int h_blockCounter = 0;
            cudaMemcpyToSymbol(
                d_blockCounter, &h_blockCounter, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

            dim3 dimBlock(16, 16);
            dim3 dimGrid(divup(imageW, dimBlock.x), divup(imageH, dimBlock.y));

            double scale = (Float)1.0 / delta;

            kernel_distest_block_scheduling<<< multiProcessorCount, dimBlock >>>(
                image, imageW, imageH, (Float)minX, (Float)maxY, (Float)delta, (Float)scale,
                maxit, dimGrid.x, dimGrid.x * dimGrid.y);
        }
        else
        {
            dim3 dimBlock(16, 16);
            dim3 dimGrid(divup(imageW, dimBlock.x), divup(imageH, dimBlock.y));

            double scale = (Float)1.0 / delta;

            kernel_distest<<< dimGrid, dimBlock >>>(
                image, imageW, imageH, (Float)minX, (Float)maxY, (Float)delta, (Float)scale, maxit);
        }
        break;

    case MANDEL_COMPARE:
        {
            dim3 dimBlock(16, 16);
            dim3 dimGrid(divup(imageW, dimBlock.x - 2), divup(imageH, dimBlock.y - 2));

            unsigned int sharedMemoryBytes = sizeof(unsigned int) * dimBlock.x * dimBlock.y;

            kernel_cn<<< dimGrid, dimBlock, sharedMemoryBytes >>>
                (image, imageW, imageH, (Float)minX, (Float)maxY, (Float)delta, maxit);
        }
        break;
    }
}

extern "C" void cuda_mandel32(
    color4*             image,
    const int           imageW,
    const int           imageH,
    const float         cenX,
    const float         cenY,
    const float         radius,
    const unsigned int  maxit,
    const int           algorithm,
    const int           blockSceduling,
    const unsigned int  multiProcessorCount)
{
    mandel(image, imageW, imageH, cenX, cenY, radius, maxit, algorithm, blockSceduling, multiProcessorCount);
}

extern "C" void cuda_mandel64(
    color4*             image,
    const int           imageW,
    const int           imageH,
    const double        cenX,
    const double        cenY,
    const double        radius,
    const unsigned int  maxit,
    const int           algorithm,
    const int           blockSceduling,
    const unsigned int  multiProcessorCount)
{
    mandel(image, imageW, imageH, cenX, cenY, radius, maxit, algorithm, blockSceduling, multiProcessorCount);
}
