#pragma once

#ifdef _MSC_VER
#define ATTR_ALIGN(N) __declspec(align(N))
#else
#define ATTR_ALIGN(N) __attribute__((aligned(N)))
#endif

struct ATTR_ALIGN(4) color4
{
    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char a;
};

static __device__ __host__ color4 make_color4(
    unsigned char b,
    unsigned char g,
    unsigned char r,
    unsigned char a = 255)
{
    color4 clr = {b,g,r,a};
    return clr;
}

#define MANDEL_LEVELSET       0
#define MANDEL_LEVELSETCONT   1
#define MANDEL_DISTEST        2
#define MANDEL_COMPARE        3

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
    const unsigned int  multiProcessorCount);

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
    const unsigned int  multiProcessorCount);
