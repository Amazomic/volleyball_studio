
typedef unsigned char uchar;

__device__ __forceinline__ int bound(int v, int extent) {
    return v < 0 ? 0 : v >= extent ? extent - 1 : v;
}
__device__ __forceinline__ float sample_y(
    const uchar* data,int pitch,int width,int height,float sx,float sy,float out_w,float out_h
) {
    float rx=fmaxf(1.f,(float)width/out_w), ry=fmaxf(1.f,(float)height/out_h);
    int start_x=(int)floorf(sx-rx), end_x=(int)ceilf(sx+rx);
    int start_y=(int)floorf(sy-ry), end_y=(int)ceilf(sy+ry);
    float sum=0.f, total=0.f;
    for(int y=start_y;y<=end_y;y++) {
        float wy=fmaxf(0.f,1.f-fabsf((float)y-sy)/ry);
        if(wy<=0.f) continue;
        for(int x=start_x;x<=end_x;x++) {
            float wx=fmaxf(0.f,1.f-fabsf((float)x-sx)/rx);
            float weight=wx*wy;
            if(weight<=0.f) continue;
            sum += (float)data[bound(y,height)*pitch+bound(x,width)]*weight;
            total += weight;
        }
    }
    return total>0.f ? sum/total : 0.f;
}
__device__ __forceinline__ float sample_uv(
    const uchar* data,int pitch,int width,int height,float sx,float sy,float rx,float ry,int component
) {
    int start_x=(int)floorf(sx-rx), end_x=(int)ceilf(sx+rx);
    int start_y=(int)floorf(sy-ry), end_y=(int)ceilf(sy+ry);
    float sum=0.f,total=0.f;
    for(int y=start_y;y<=end_y;y++) {
        float wy=fmaxf(0.f,1.f-fabsf((float)y-sy)/ry);
        if(wy<=0.f) continue;
        for(int x=start_x;x<=end_x;x++) {
            float wx=fmaxf(0.f,1.f-fabsf((float)x-sx)/rx);
            float weight=wx*wy;
            if(weight<=0.f) continue;
            sum += (float)data[bound(y,height)*pitch+bound(x,width)*2+component]*weight;
            total += weight;
        }
    }
    return total>0.f ? sum/total : 128.f;
}
__device__ __forceinline__ float rgb_pixel(const uchar* src,int pitch,int width,int height,int x,int y,int channel,int matrix) {
    if(x<0 || y<0 || x>=width || y>=height) return 0.f;
    const uchar *uv=src+pitch*height;
    float yy=(float)src[y*pitch+x] - 16.f;
    int cx=x/2, cy=y/2;
    float u=(float)uv[cy*pitch+cx*2]-128.f;
    float v=(float)uv[cy*pitch+cx*2+1]-128.f;
    float r=1.16438356f*yy+(matrix==1 ? 1.79274107f:matrix==2 ? 1.67867f:1.59602678f)*v;
    float g=1.16438356f*yy-(matrix==1 ? 0.21324861f:matrix==2 ? 0.187326f:0.39176229f)*u-(matrix==1 ? 0.53290933f:matrix==2 ? 0.650424f:0.81296765f)*v;
    float b=1.16438356f*yy+(matrix==1 ? 2.11240179f:matrix==2 ? 2.14177f:2.01723214f)*u;
    float value= channel==0?r:channel==1?g:b;
    return floorf(fminf(fmaxf(value,0.f),255.f)+0.5f);
}
__device__ __forceinline__ float sample_rgb(
    const uchar* src,int pitch,int width,int height,float sx,float sy,int channel,int matrix
) {
    float yy=sample_y(src,pitch,width,height,sx,sy,640.f,360.f)-16.f;
    const uchar* uv=src+pitch*height;
    float rx=fmaxf(1.f,(float)width/640.f),ry=fmaxf(1.f,(float)height/360.f);
    float ux=(sx-0.5f)*0.5f,uy=(sy-0.5f)*0.5f;
    float u=sample_uv(uv,pitch,width/2,height/2,ux,uy,rx,ry,0)-128.f;
    float v=sample_uv(uv,pitch,width/2,height/2,ux,uy,rx,ry,1)-128.f;
    float r=1.16438356f*yy+(matrix==1 ? 1.79274107f:matrix==2 ? 1.67867f:1.59602678f)*v;
    float g=1.16438356f*yy-(matrix==1 ? 0.21324861f:matrix==2 ? 0.187326f:0.39176229f)*u-(matrix==1 ? 0.53290933f:matrix==2 ? 0.650424f:0.81296765f)*v;
    float b=1.16438356f*yy+(matrix==1 ? 2.11240179f:matrix==2 ? 2.14177f:2.01723214f)*u;
    return channel==0?r:channel==1?g:b;
}
extern "C" __global__ void ball_gray(
    const uchar* src,int pitch,int width,int height,float* dst,int slot
) {
    int x=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int y=(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(x>=512||y>=288) return;
    float sx=((float)x+0.5f)*(float)width/512.f-0.5f;
    float sy=((float)y+0.5f)*(float)height/288.f-0.5f;
    float luma=sample_y(src,pitch,width,height,sx,sy,512.f,288.f);
    dst[slot*512*288+y*512+x]=fminf(fmaxf((luma-16.f)/219.f,0.f),1.f);
}
extern "C" __global__ void rgb_640(
    const uchar* src,int pitch,int width,int height,uchar* dst,int matrix
) {
    int x=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int y=(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(x>=640||y>=360) return;
    float sx=((float)x+0.5f)*(float)width/640.f-0.5f;
    float sy=((float)y+0.5f)*(float)height/360.f-0.5f;
    for(int c=0;c<3;c++) {
        float val=sample_rgb(src,pitch,width,height,sx,sy,c,matrix);
        dst[(y*640+x)*3+c]=(uchar)(fminf(fmaxf(floorf(val+0.5f),0.f),255.f));
    }
}
extern "C" __global__ void rfdetr_image(const uchar* rgb,float* dst) {
    int x=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int y=(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(x>=576||y>=576) return;
    float xx=fminf(fmaxf(((float)x+0.5f)*640.f/576.f-0.5f,0.f),639.f);
    float yy=fminf(fmaxf(((float)y+0.5f)*360.f/576.f-0.5f,0.f),359.f);
    int x0=(int)floorf(xx),y0=(int)floorf(yy);
    int x1=bound(x0+1,640),y1=bound(y0+1,360);
    float u=xx-x0,v=yy-y0;
    const float mean[3]={0.485f,0.456f,0.406f};
    const float stds[3]={0.229f,0.224f,0.225f};
    for(int c=0;c<3;c++) {
        float a=rgb[(y0*640+x0)*3+c],b=rgb[(y0*640+x1)*3+c];
        float d=rgb[(y1*640+x0)*3+c],e=rgb[(y1*640+x1)*3+c];
        float z=(a+(b-a)*u)*(1.f-v)+(d+(e-d)*u)*v;
        dst[c*576*576+y*576+x]=(floorf(fminf(fmaxf(z,0.f),255.f)+0.5f)/255.f-mean[c])/stds[c];
    }
}
extern "C" __global__ void pose_crop(
    const uchar* src,int pitch,int width,int height,const float* boxes,float* dst,int matrix
) {
    int x=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int y=(int)(blockIdx.y*blockDim.y+threadIdx.y);
    int crop=(int)blockIdx.z;
    if(x>=192||y>=256) return;
    float cx=boxes[crop*4+0],cy=boxes[crop*4+1];
    float crop_w=boxes[crop*4+2],crop_h=boxes[crop*4+3];
    float fx=cx+(((float)x+0.5f)/192.f-0.5f)*crop_w-0.5f;
    float fy=cy+(((float)y+0.5f)/256.f-0.5f)*crop_h-0.5f;
    int x0=(int)floorf(fx),y0=(int)floorf(fy);
    float u=fx-(float)x0,v=fy-(float)y0;
    const float mean[3]={123.675f,116.28f,103.53f};
    const float stds[3]={58.395f,57.12f,57.375f};
    for(int c=0;c<3;c++) {
        float a=rgb_pixel(src,pitch,width,height,x0,y0,c,matrix);
        float b=rgb_pixel(src,pitch,width,height,x0+1,y0,c,matrix);
        float d=rgb_pixel(src,pitch,width,height,x0,y0+1,c,matrix);
        float e=rgb_pixel(src,pitch,width,height,x0+1,y0+1,c,matrix);
        float pixel=(a+(b-a)*u)*(1.f-v)+(d+(e-d)*u)*v;
        dst[(crop*3+c)*256*192+y*192+x]=(pixel-mean[c])/stds[c];
    }
}


__device__ __forceinline__ int normalized_luma(const uchar* src,int pitch,int x,int y,int depth,int full_range) {
    int val=depth==8 ? (int)src[y*pitch+x] :
        (int)(*((const unsigned short*)(src+y*pitch)+x)>>8);
    if (full_range) val=(int)floorf(16.f+(float)val*219.f/255.f+0.5f);
    return bound(val,256);
}
__device__ __forceinline__ int normalized_chroma(const uchar* uv,int pitch,int x,int y,int depth,int full_range) {
    int val=depth==8 ? (int)uv[y*pitch+x] :
        (int)(*((const unsigned short*)(uv+y*pitch)+x)>>8);
    if (full_range) val=(int)floorf(128.f+((float)val-128.f)*224.f/255.f+0.5f);
    return bound(val,256);
}
__device__ __forceinline__ void map_rotation(int x,int y,int sw,int sh,int rotation,int* sx,int* sy) {
    if(rotation==90){*sx=sw-1-y;*sy=x;}
    else if(rotation==180){*sx=sw-1-x;*sy=sh-1-y;}
    else if(rotation==270){*sx=y;*sy=sh-1-x;}
    else {*sx=x;*sy=y;}
}
extern "C" __global__ void canonical_nv12(
    const uchar* src,int pitch,int sw,int sh,uchar* dst,int ow,int oh,int rotation,int depth,int chroma,int full_range
) {
    int x=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int y=(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(x>=ow||y>=oh)return;
    int sx,sy;
    map_rotation(x,y,sw,sh,rotation,&sx,&sy);
    dst[y*ow+x]=(uchar)normalized_luma(src,pitch,sx,sy,depth,full_range);
    if((x&1)==0 && (y&1)==0) {
        const uchar* uv=src+pitch*sh;
        int cx=(sx/2)*2, cy=chroma==1 ? sy/2 : sy;
        uchar* out_uv=dst+ow*oh+(y/2)*ow+x;
        if(chroma==3) {
            const uchar* v=uv+pitch*sh;
            int off=sx;
            out_uv[0]=(uchar)normalized_chroma(uv,pitch,off,sy,depth,full_range);
            out_uv[1]=(uchar)normalized_chroma(v,pitch,off,sy,depth,full_range);
        } else {
            out_uv[0]=(uchar)normalized_chroma(uv,pitch,cx,cy,depth,full_range);
            out_uv[1]=(uchar)normalized_chroma(uv,pitch,cx+1,cy,depth,full_range);
        }
    }
}
