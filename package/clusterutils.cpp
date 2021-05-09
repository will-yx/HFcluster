#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <random>

#define SQ(x) ((x)*(x))
#define MIN(a,b) ((a)<=(b)?(a):(b))
#define MAX(a,b) ((a)>=(b)?(a):(b))

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

extern "C" {

void fixed_vs_fixed_count(float *niche_acc, float *mindist_acc, const int *data, const int *firstidx, int nc, int n, int nr) {
  const int nr2 = SQ(nr);
  
  int *xs = (int *) malloc(n * sizeof(int));
  int *ys = (int *) malloc(n * sizeof(int));
  
  for(int d=0; d<n; d++) {
    xs[d] = data[d*4 + 2];
    ys[d] = data[d*4 + 3];
  }
  
  for(int c1=0; c1 < nc; c1++) { // for each cluster type for the origin
    for(int c2=0; c2 < nc; c2++) { // for every cluster type neighboring i1
      
      if(c1==c2) {
        for(int i1=firstidx[c1]; i1 < firstidx[c1+1]; i1++) { // for every event of this cluster type (as origin for the neighborhood)
          int count = 0; // count how many events of this cluster type are within the radius
          int mind2 = INT_MAX;
          for(int i2=firstidx[c2]; i2 < firstidx[c2+1]; i2++) {
            if(i2==i1) continue;
            int d2 = SQ(xs[i2] - xs[i1]) + SQ(ys[i2] - ys[i1]);
            if(d2 <= nr2) count++;
            if(d2 < mind2) mind2 = d2;
          }
          niche_acc[i1 * nc + c2] += count;
          if(mind2 < INT_MAX) mindist_acc[i1 * nc + c2] += sqrt(mind2);
        }
      } else {
        for(int i1=firstidx[c1]; i1 < firstidx[c1+1]; i1++) { // for every event of this cluster type (as origin for the neighborhood)
          int count = 0; // count how many events of this cluster type are within the radius
          int mind2 = INT_MAX;
          for(int i2=firstidx[c2]; i2 < firstidx[c2+1]; i2++) {
            int d2 = SQ(xs[i2] - xs[i1]) + SQ(ys[i2] - ys[i1]);
            if(d2 <= nr2) count++;
            if(d2 < mind2) mind2 = d2;
          }
          niche_acc[i1 * nc + c2] += count;
          if(mind2 < INT_MAX) mindist_acc[i1 * nc + c2] += sqrt(mind2);
        }
      }
      
    }
  }
  
  free(xs); free(ys);
}

void fixed_origin_random_neighbors(float *niche_acc, float *mindist_acc, const int *data, const int *firstidx, const uint8 *valid, int ds, int nc, int n, int w, int h, int nr, int cd, int min_count, int it) {
  long unsigned int seed = static_cast<long unsigned int>(time(0)) + it;
  std::default_random_engine generator(seed);
  
  std::uniform_int_distribution<int> random_x(0, w-1);
  std::uniform_int_distribution<int> random_y(0, h-1);
  
  const int nr2 = SQ(nr);
  const int max_r2 = SQ(w) + SQ(h);
  
  const int wds = 1 + (w-1) / ds;
  const int hds = 1 + (h-1) / ds;
  
  const int cd_divisor = cd / 3;
  
  const int w_3 = 1 + (w-1) / cd_divisor;
  const int h_3 = 1 + (h-1) / cd_divisor;
  
  const int wo = w_3 + 4;
  const int ho = h_3 + 4;
  
  int *count = (int *) malloc(nc * sizeof(int));
  int *loops = (int *) malloc(nc * sizeof(int));
  
  for(int c=0; c<nc; c++) {
    count[c] = firstidx[c+1] - firstidx[c];
    loops[c] = 1 + (min_count - 1) / count[c];
  }
  
  int *x1s = (int *) malloc(n * sizeof(int));
  int *y1s = (int *) malloc(n * sizeof(int));
  
  int *x2s = (int *) malloc(n * sizeof(int));
  int *y2s = (int *) malloc(n * sizeof(int));
  
  uint8 *occupied_padded = (uint8 *) malloc(wo*ho* sizeof(uint8));
  if(!occupied_padded) printf(" ! Error: couldn't allocate %.2f MB\n", ((double)wo*ho)/(1000*1000));
  
  uint8 *occupied_padded_c1 = (uint8 *) malloc(wo*ho* sizeof(uint8));
  if(!occupied_padded_c1) printf(" ! Error: couldn't allocate %.2f MB\n", ((double)wo*ho)/(1000*1000));
  
  uint8 *occupied = &occupied_padded[2*wo+2];
  uint8 *occupied_c1 = &occupied_padded_c1[2*wo+2];
  
  for(int d=0; d<n; d++) {
    x1s[d] = data[d*4 + 2];
    y1s[d] = data[d*4 + 3];
  }
  
  for(int c1=0; c1 < nc; c1++) { // for every cluster type as origin
    memset(occupied_padded_c1, 0, wo*ho * sizeof(uint8));
    
    for(int i=firstidx[c1]; i < firstidx[c1+1]; i++) { // fill occupied with real origin events
      int x_3 = x1s[i] / cd_divisor;
      int y_3 = y1s[i] / cd_divisor;
      uint8 *o = &occupied_c1[y_3 * wo + x_3];
      memset(&o[-wo-wo-1], 1, 3 * sizeof(uint8));
      memset(&o[   -wo-2], 1, 5 * sizeof(uint8));
      memset(&o[      -2], 1, 5 * sizeof(uint8));
      memset(&o[    wo-2], 1, 5 * sizeof(uint8));
      memset(&o[ wo+wo-1], 1, 3 * sizeof(uint8));
    }
    
    for(int c2=0; c2 < nc; c2++) { // for every cluster type as the neighbors
      for(int loop2=0; loop2 < loops[c2]; loop2++) {
        memcpy(occupied_padded, occupied_padded_c1, wo*ho * sizeof(uint8));
        
        for(int i2=firstidx[c2]; i2 < firstidx[c2+1];) { // generate random neighbor points that don't conflict pre-existing points
          int x = random_x(generator);
          int y = random_y(generator);
          int xds = x / ds;
          int yds = y / ds;
          int x_3 = x / cd_divisor;
          int y_3 = y / cd_divisor;
          uint8 *o = &occupied[y_3 * wo + x_3];
          if(valid[yds*wds + xds] && !o[0]) {
            memset(&o[-wo-wo-1], 1, 3 * sizeof(uint8));
            memset(&o[   -wo-2], 1, 5 * sizeof(uint8));
            memset(&o[      -2], 1, 5 * sizeof(uint8));
            memset(&o[    wo-2], 1, 5 * sizeof(uint8));
            memset(&o[ wo+wo-1], 1, 3 * sizeof(uint8));
            x2s[i2] = x;
            y2s[i2] = y;
            i2++;
          }
        }
        if(c2==c1) {
          x2s[firstidx[c2]] = -w;
          y2s[firstidx[c2]] = -h;
        }
        
        for(int i1=firstidx[c1]; i1 < firstidx[c1+1]; i1++) { // for every origin event of type c1
          int x1 = x1s[i1];
          int y1 = y1s[i1];
          
          int niche_count = 0; // count how many events of type c2 are within the radius
          int mind2 = INT_MAX;
          
          for(int i2 = firstidx[c2]; i2 < firstidx[c2+1]; i2++) {
            int x2 = x2s[i2];
            int y2 = y2s[i2];
            int d2 = SQ(x2-x1) + SQ(y2-y1);
            if(d2 <= nr2) niche_count++;
            if(d2 < mind2) mind2 = d2;
          }
          niche_acc[i1 * nc + c2] += (float) niche_count / loops[c2];
          if(mind2 < max_r2) mindist_acc[i1 * nc + c2] += sqrt(mind2) / loops[c2];
        }
      }
    }
  }
  
  free(x1s); free(y1s);
  free(x2s); free(y2s);
  free(occupied_padded);
  free(occupied_padded_c1);
  free(count);
  free(loops);
}

void random_origin_fixed_neighbors(float *niche_acc, float *mindist_acc, const int *data, const int *firstidx, const uint8 *valid, int ds, int nc, int n, int w, int h, int nr, int cd, int min_count, int it) {
  long unsigned int seed = static_cast<long unsigned int>(time(0)) + it;
  std::default_random_engine generator(seed);
  
  std::uniform_int_distribution<int> random_x(0, w-1);
  std::uniform_int_distribution<int> random_y(0, h-1);
  
  const int nr2 = SQ(nr);
  const int max_r2 = SQ(w) + SQ(h);
  
  const int wds = 1 + (w-1) / ds;
  const int hds = 1 + (h-1) / ds;
  
  const int cd_divisor = cd / 3;
  
  const int w_3 = 1 + (w-1) / cd_divisor;
  const int h_3 = 1 + (h-1) / cd_divisor;
  
  const int wo = w_3 + 4;
  const int ho = h_3 + 4;
  
  int *count = (int *) malloc(nc * sizeof(int));
  int *loops = (int *) malloc(nc * sizeof(int));
  
  for(int c=0; c<nc; c++) {
    count[c] = firstidx[c+1] - firstidx[c];
    loops[c] = 1 + (min_count - 1) / count[c];
  }
  
  int *x1s = (int *) malloc(n * sizeof(int));
  int *y1s = (int *) malloc(n * sizeof(int));
  
  int *x2s = (int *) malloc(n * sizeof(int));
  int *y2s = (int *) malloc(n * sizeof(int));
  
  uint8 *occupied_padded = (uint8 *) malloc(wo*ho* sizeof(uint8));
  if(!occupied_padded) printf(" ! Error: couldn't allocate %.2f MB\n", ((double)wo*ho)/(1000*1000));
  
  uint8 *occupied_padded_c2 = (uint8 *) malloc(wo*ho* sizeof(uint8));
  if(!occupied_padded_c2) printf(" ! Error: couldn't allocate %.2f MB\n", ((double)wo*ho)/(1000*1000));
  
  uint8 *occupied = &occupied_padded[2*wo+2];
  uint8 *occupied_c2 = &occupied_padded_c2[2*wo+2];
  
  for(int d=0; d<n; d++) {
    x2s[d] = data[d*4 + 2];
    y2s[d] = data[d*4 + 3];
  }
  
  for(int c2=0; c2 < nc; c2++) { // for every cluster type neighboring i1
    memset(occupied_padded_c2, 0, wo*ho * sizeof(uint8));
    
    for(int i=firstidx[c2]; i < firstidx[c2+1]; i++) { // fill occupied with fixed neighbors
      int x_3 = x2s[i] / cd_divisor;
      int y_3 = y2s[i] / cd_divisor;
      uint8 *o = &occupied_c2[y_3 * wo + x_3];
      memset(&o[-wo-wo-1], 1, 3 * sizeof(uint8));
      memset(&o[   -wo-2], 1, 5 * sizeof(uint8));
      memset(&o[      -2], 1, 5 * sizeof(uint8));
      memset(&o[    wo-2], 1, 5 * sizeof(uint8));
      memset(&o[ wo+wo-1], 1, 3 * sizeof(uint8));
    }
    
    for(int c1=0; c1 < nc; c1++) { // for every cluster type as the origin
      for(int loop1=0; loop1 < loops[c1]; loop1++) {
        memcpy(occupied_padded, occupied_padded_c2, wo*ho * sizeof(uint8));
        
        for(int i1=firstidx[c1]; i1 < firstidx[c1+1];) {
          int x = random_x(generator);
          int y = random_y(generator);
          int xds = x / ds;
          int yds = y / ds;
          int x_3 = x / cd_divisor;
          int y_3 = y / cd_divisor;
          uint8 *o = &occupied[y_3 * wo + x_3];
          if(valid[yds*wds + xds] && !o[0]) {
            memset(&o[-wo-wo-1], 1, 3 * sizeof(uint8));
            memset(&o[   -wo-2], 1, 5 * sizeof(uint8));
            memset(&o[      -2], 1, 5 * sizeof(uint8));
            memset(&o[    wo-2], 1, 5 * sizeof(uint8));
            memset(&o[ wo+wo-1], 1, 3 * sizeof(uint8));
            x1s[i1] = x;
            y1s[i1] = y;
            i1++;
          }
        }
        if(c2==c1) {
          x1s[firstidx[c1]] = -w;
          y1s[firstidx[c1]] = -h;
        }
        
        for(int i1=firstidx[c1]; i1 < firstidx[c1+1]; i1++) { // for every event (as origin for the neighborhood)
          int x1 = x1s[i1];
          int y1 = y1s[i1];
          
          int niche_count = 0; // count how many events of this cluster type are within the radius
          int mind2 = INT_MAX;
          for(int i2 = firstidx[c2]; i2 < firstidx[c2+1]; i2++) {
            int x2 = x2s[i2];
            int y2 = y2s[i2];
            int d2 = SQ(x2-x1) + SQ(y2-y1);
            if(d2 <= nr2) niche_count++;
            if(d2 < mind2) mind2 = d2;
          }
          niche_acc[i1 * nc + c2] += (float) niche_count / loops[c1];
          if(mind2 < max_r2) mindist_acc[i1 * nc + c2] += sqrt(mind2) / loops[c1];
        }
      }
    }
  }
  
  free(x1s); free(y1s);
  free(x2s); free(y2s);
  free(occupied_padded);
  free(occupied_padded_c2);
  free(count);
  free(loops);
}

void fixed_origin_probabilities(float *niche_acc, float *mindist_acc, const int *data, const int *firstidx, int nc, int n, int nr) {
  const int nr2 = SQ(nr);
  
  int *count = (int *) malloc(nc * sizeof(int));
  
  for(int c=0; c<nc; c++) count[c] = firstidx[c+1] - firstidx[c];
  
  int *xs = (int *) malloc(n * sizeof(int));
  int *ys = (int *) malloc(n * sizeof(int));
  double *dist = (double *) malloc(n * sizeof(double));
  
  double *ptable = (double *) malloc(nc*n * sizeof(double));
  
  for(int d=0; d<n; d++) {
    xs[d] = data[d*4 + 2];
    ys[d] = data[d*4 + 3];
  }
  
  for(int c1=0; c1 < nc; c1++) { // for every cluster type as the origin
    memset(ptable, 0, nc*n * sizeof(double));
    for(int c2=0; c2 < nc; c2++) { // for every cluster c2
      double cnt2 = count[c2] - (c2==c1 ? 1 : 0);
      double p = cnt2 > 0 ? 1 : 0;
      double d = 0;
      for(int i2=0; i2 < n-1 && p > 1e-3; i2++) {
        double p2 = cnt2 / (n-1-i2);
        ptable[c2*n + i2] =  p * p2;
        p *= (1.0f - p2);
      }
    }
    
    for(int i1=firstidx[c1]; i1 < firstidx[c1+1]; i1++) { // for every event (as origin for the neighborhood)
      int x1 = xs[i1];
      int y1 = ys[i1];
      
      int niche_count = 0;
      for(int i2=0; i2 < n; i2++) { // for every other cell
        if(i2==i1) continue;
        int x2 = xs[i2];
        int y2 = ys[i2];
        int d2 = SQ(x2-x1) + SQ(y2-y1);
        dist[i2] = sqrt(d2);
        if(d2 <= nr2) niche_count++;
      }
      // there are niche_count cells within niche radius
      // for example, 10 cells within nr.  imagine there are 3 populations
      // count[0] = 2
      // count[1] = 10
      // count[2] = 100
      // if there are 10 cells near cell{0}, we expect:
      // p{2} = 100 * 10 / (100+10+2-1) ~ 9
      // p{1} = 10 * 10 / (100+10+2-1) ~ 0.9
      // p{0} = (2-1) * 10 / (n-1) ~ 0.09
      
      dist[i1] = dist[n-1];
      std::sort(dist, &dist[n-1]);
      
      for(int c2=0; c2 < nc; c2++) { // for every cluster c2
        double d = 0;
        for(int i2=0; i2 < n-1; i2++) {
          float p = ptable[c2*n + i2];
          d += dist[i2] * p;
          if(p < 1e-6) break;
        }
        mindist_acc[i1 * nc + c2] += d;
        niche_acc[i1 * nc + c2] += (float) (niche_count * (count[c2] - (c2==c1 ? 1 : 0))) / (n-1);
      }
    }
  }
  free(xs); free(ys);
  free(dist);
  free(ptable);
  free(count);
}

}


