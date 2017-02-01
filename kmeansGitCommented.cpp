/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

////////////////////////////////////////// kmeans ////////////////////////////////////////////

namespace cv
{

// generating random center
static void generateRandomCenter(const std::vector<Vec2f>& box, float* center, RNG& rng)
{
    size_t j, dims = box.size();
    float margin = 1.f/dims;
    for( j = 0; j < dims; j++ )
        center[j] = ((float)rng*(1.f+margin*2.f)-margin)*(box[j][1] - box[j][0]) + box[j][0];
}

class KMeansPPDistanceComputer : public ParallelLoopBody
{
public:
    KMeansPPDistanceComputer( float *_tdist2,
                              const float *_data,
                              const float *_dist,
                              int _dims,
                              size_t _step,
                              size_t _stepci )
        : tdist2(_tdist2),
          data(_data),
          dist(_dist),
          dims(_dims),
          step(_step),
          stepci(_stepci) { }

    void operator()( const cv::Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;

        for ( int i = begin; i<end; i++ )
        {
            tdist2[i] = std::min(normL2Sqr(data + step*i, data + stepci, dims), dist[i]);
        }
    }

private:
    KMeansPPDistanceComputer& operator=(const KMeansPPDistanceComputer&); // to quiet MSVC

    float *tdist2;
    const float *data;
    const float *dist;
    const int dims;
    const size_t step;
    const size_t stepci;
};

/*
k-means center initialization using the following algorithm:
Arthur & Vassilvitskii (2007) k-means++: The Advantages of Careful Seeding
*/
static void generateCentersPP(const Mat& _data, Mat& _out_centers,
                              int K, RNG& rng, int trials)
{
    int i, j, k, dims = _data.cols, N = _data.rows;
    const float* data = _data.ptr<float>(0);
    size_t step = _data.step/sizeof(data[0]);
    std::vector<int> _centers(K);
    int* centers = &_centers[0];
    std::vector<float> _dist(N*3);
    float* dist = &_dist[0], *tdist = dist + N, *tdist2 = tdist + N;
    double sum0 = 0;

    centers[0] = (unsigned)rng % N;

    for( i = 0; i < N; i++ )
    {
        dist[i] = normL2Sqr(data + step*i, data + step*centers[0], dims);
        sum0 += dist[i];
    }

    for( k = 1; k < K; k++ )
    {
        double bestSum = DBL_MAX;
        int bestCenter = -1;

        for( j = 0; j < trials; j++ )
        {
            double p = (double)rng*sum0, s = 0;
            for( i = 0; i < N-1; i++ )
                if( (p -= dist[i]) <= 0 )
                    break;
            int ci = i;

            parallel_for_(Range(0, N),
                         KMeansPPDistanceComputer(tdist2, data, dist, dims, step, step*ci));
            for( i = 0; i < N; i++ )
            {
                s += tdist2[i];
            }

            if( s < bestSum )
            {
                bestSum = s;
                bestCenter = ci;
                std::swap(tdist, tdist2);
            }
        }
        centers[k] = bestCenter;
        sum0 = bestSum;
        std::swap(dist, tdist);
    }

    for( k = 0; k < K; k++ )
    {
        const float* src = data + step*centers[k];
        float* dst = _out_centers.ptr<float>(k);
        for( j = 0; j < dims; j++ )
            dst[j] = src[j];
    }
}

class KMeansDistanceComputer : public ParallelLoopBody
{
public:
    KMeansDistanceComputer( double *_distances,
                            int *_labels,
                            const Mat& _data,
                            const Mat& _centers )
        : distances(_distances),
          labels(_labels),
          data(_data),
          centers(_centers)
    {
    }

    void operator()( const Range& range ) const
    {
        const int begin = range.start;
        const int end = range.end;
        const int K = centers.rows;
        const int dims = centers.cols;

        for( int i = begin; i<end; ++i)
        {
            const float *sample = data.ptr<float>(i);
            int k_best = 0;
            double min_dist = DBL_MAX;

            for( int k = 0; k < K; k++ )
            {
                const float* center = centers.ptr<float>(k);
                const double dist = normL2Sqr(sample, center, dims);

                if( min_dist > dist )
                {
                    min_dist = dist;
                    k_best = k;
                }
            }

            distances[i] = min_dist;
            labels[i] = k_best;
        }
    }

private:
    KMeansDistanceComputer& operator=(const KMeansDistanceComputer&); // to quiet MSVC

    double *distances;
    int *labels;
    const Mat& data;
    const Mat& centers;
};

}

double cv::kmeans( InputArray _data, int K,
                   InputOutputArray _bestLabels,
                   TermCriteria criteria, int attempts,
                   int flags, OutputArray _centers )
{
    CV_INSTRUMENT_REGION() // apa ini :(

    const int SPP_TRIALS = 3;

    Mat data0 = _data.getMat();  // return matrix header
    bool isrow = data0.rows == 1; // check whether the row of data0 is 1
    int N = isrow ? data0.cols : data0.rows; // N adalah jumlah data (entah kolom atau row)
    //.channels will return the number of matrix channels. Calculate the dims. If isrow==TRUE, then dims=1*data0.channels. 
    // if isrow==FALSE, then dims = data0.cols*data0.channels
    int dims = (isrow ? 1 : data0.cols)*data0.channels(); // dims = data0.channels*1 , nyimpen channel
    int type = data0.depth(); // return the depth. . Such as 16S (16bit signed) or 8U (8unsigned)
    attempts = std::max(attempts, 1); // just compare it with 1 to check if attempts<1, then attempts = 1
    
    // if the conditions are not satisfied, terminate program
    CV_Assert( data0.dims <= 2 && type == CV_32F && K > 0 );
    CV_Assert( N >= K );

    // create data (mat-type) with Nxdims, type CV_32F, use ptr of data0, and step like condition above
    // step : number of bytes each matrix row occupies
    Mat data(N, dims, CV_32F, data0.ptr(), isrow ? dims * sizeof(float) : static_cast<size_t>(data0.step));

    // create bestLabels to pass Mat to function with Nx1 , type CV_32S, -1 and allow transposed
    _bestLabels.create(N, 1, CV_32S, -1, true);

    Mat _labels, best_labels = _bestLabels.getMat(); // create labels, then best_labels = getMat from _bestLabels

    // CONDITION
    // flags & CV_KMEANS_USE... = 1
    if( flags & CV_KMEANS_USE_INITIAL_LABELS )
    {
        // this must be satisfied!
        CV_Assert( (best_labels.cols == 1 || best_labels.rows == 1) &&
                  best_labels.cols*best_labels.rows == N &&
                  best_labels.type() == CV_32S &&
                  best_labels.isContinuous());
        best_labels.copyTo(_labels); // best_labels data matrix are copied to labels
    }
    else // kondisi lain
    {
        // jika tidak sesuai syarat colom atau label = 1
        if( !((best_labels.cols == 1 || best_labels.rows == 1) &&
             best_labels.cols*best_labels.rows == N &&
            best_labels.type() == CV_32S &&
            best_labels.isContinuous()))
            best_labels.create(N, 1, CV_32S); // buat best_labels dengan row N, column 1, type CV_32S

        _labels.create(best_labels.size(), best_labels.type()); // create _labels with size and type = best_labels's size and type
    }

    int* labels = _labels.ptr<int>(); // I DONT KNOW WHAT THE FUCK IS THIS

    // membuat centers, old centers dan temporary
    // Centers dan old centers memiliki K-baris, temp cm punya 1 
    Mat centers(K, dims, type), old_centers(K, dims, type), temp(1, dims, type);

    std::vector<int> counters(K); // buat vector counters bertipe int dengan kapasitas-K

    // buat vector _box bertipe Vec2f dengan kapasitas dims
    // vector 2f: vector dengan tipe float dan alloc 2 dengan kapasitas dims
    std::vector<Vec2f> _box(dims); 

    Vec2f* box = &_box[0]; // buat pointer thdp elemen pertama vector _box

    // buat variable best_compactness = maximal double, dan compactness = 0
    double best_compactness = DBL_MAX, compactness = 0; 

    RNG& rng = theRNG(); // dapatkan nilai random dari RNG
    int a, iter, i, j, k; // deklarasikan a,iter,i,j,k

    // PENENTUAN KRITERIA
    // Apa itu criteria, apa itu epsilon, apa itu EPS?
    if( criteria.type & TermCriteria::EPS ) 
        criteria.epsilon = std::max(criteria.epsilon, 0.);
    else
        criteria.epsilon = FLT_EPSILON;
    criteria.epsilon *= criteria.epsilon;

    if( criteria.type & TermCriteria::COUNT )
        criteria.maxCount = std::min(std::max(criteria.maxCount, 2), 100);
    else
        criteria.maxCount = 100;

    // jika K = 1
    if( K == 1 )
    {
        attempts = 1;
        criteria.maxCount = 2;
    }

    // buat pointer float ke data
    const float* sample = data.ptr<float>(0);
    for( j = 0; j < dims; j++ )
        box[j] = Vec2f(sample[j], sample[j]);

    for( i = 1; i < N; i++ )
    {
        sample = data.ptr<float>(i);
        for( j = 0; j < dims; j++ )
        {
            float v = sample[j];
            box[j][0] = std::min(box[j][0], v);
            box[j][1] = std::max(box[j][1], v);
        }
    }

    // KOMEN SAMPAI SINI!
    for( a = 0; a < attempts; a++ ) // untuk setiap attempt
    {
        double max_center_shift = DBL_MAX; // memiliki nilai maksimal dari double
        for( iter = 0;; ) // iter = 0
        {
            swap(centers, old_centers); // tukar centers dengan old centers

			// di tiap attempt setelah attempt pertama dan jika use initial labels
            if( iter == 0 && (a > 0 || !(flags & KMEANS_USE_INITIAL_LABELS)) ) 
            {
                if( flags & KMEANS_PP_CENTERS ) // menggunakan KMEANS PLUS PLUS
                    generateCentersPP(data, centers, K, rng, SPP_TRIALS); // hasilkan center dengan kmeans++
                else
                {
                    for( k = 0; k < K; k++ )
                        generateRandomCenter(_box, centers.ptr<float>(k), rng); // menggunakan nilai random biasa
                }
            }
            else // untuk attempt pertama 
            {
            	// untuk iter pertama dan attempt pertama dan jika use initial labels
                if( iter == 0 && a == 0 && (flags & KMEANS_USE_INITIAL_LABELS) ) 
                {
                    for( i = 0; i < N; i++ ) // N adalah jumlah data, untuk tiap data
                        CV_Assert( (unsigned)labels[i] < (unsigned)K ); // pastikan label dibawah K
                }

                // compute centers
                centers = Scalar(0); // kosongkan nilai
                for( k = 0; k < K; k++ )
                    counters[k] = 0; // kosongkan nilai vector counters yang kapasitasnya K

                for( i = 0; i < N; i++ ) // untuk tiap data
                {
                    sample = data.ptr<float>(i); // sample yang berupa konstanta pointer menyimpan pointer dengan nilai float
                    k = labels[i]; // k menyimpan labels di data i
                    float* center = centers.ptr<float>(k); // pointer cetner menyimpan nilai centers di k
                    j=0; // set j = 0
                    #if CV_ENABLE_UNROLLED // apa ini ?
                    for(; j <= dims - 4; j += 4 ) // 
                    {
                        float t0 = center[j] + sample[j];
                        float t1 = center[j+1] + sample[j+1];

                        center[j] = t0;
                        center[j+1] = t1;

                        t0 = center[j+2] + sample[j+2];
                        t1 = center[j+3] + sample[j+3];

                        center[j+2] = t0;
                        center[j+3] = t1;
                    }
                    #endif
                    for( ; j < dims; j++ ) // selama j kurang dari channel
                        center[j] += sample[j]; // hitung jumlah  dari tiap channel
                    counters[k]++; // counters menghitung jumlah dari tiap label 
                }

                if( iter > 0 ) // jika iter diatas 0 kosongkan max_center_shift
                    max_center_shift = 0; 

                for( k = 0; k < K; k++ ) // untuk setiap label k
                {
                    if( counters[k] != 0 ) // jika jumlah kelas tidak bernilai 0 lanjutkan
                        continue;

                    // if some cluster appeared to be empty then:
                    //   1. find the biggest cluster
                    //   2. find the farthest from the center point in the biggest cluster
                    //   3. exclude the farthest point from the biggest cluster and form a new 1-point cluster.
                    int max_k = 0; // variable penyimpan nilai maksimal
                    for( int k1 = 1; k1 < K; k1++ ) // untuk tiap cluster cari kluster terbesar ! (1)
                    {
                        if( counters[max_k] < counters[k1] )
                            max_k = k1;
                    }

                    double max_dist = 0;
                    int farthest_i = -1;
                    float* new_center = centers.ptr<float>(k);
                    float* old_center = centers.ptr<float>(max_k);
                    float* _old_center = temp.ptr<float>(); // normalized
                    float scale = 1.f/counters[max_k];
                    for( j = 0; j < dims; j++ )
                        _old_center[j] = old_center[j]*scale;

                    for( i = 0; i < N; i++ )
                    {
                        if( labels[i] != max_k )
                            continue;
                        sample = data.ptr<float>(i);
                        double dist = normL2Sqr(sample, _old_center, dims); // compute euclidean distance

                        if( max_dist <= dist )
                        {
                            max_dist = dist;
                            farthest_i = i;
                        }
                    }

                    counters[max_k]--;
                    counters[k]++;
                    labels[farthest_i] = k;
                    sample = data.ptr<float>(farthest_i);

                    for( j = 0; j < dims; j++ )
                    {
                        old_center[j] -= sample[j];
                        new_center[j] += sample[j];
                    }
                }

                for( k = 0; k < K; k++ )
                {
                    float* center = centers.ptr<float>(k);
                    CV_Assert( counters[k] != 0 );

                    float scale = 1.f/counters[k];
                    for( j = 0; j < dims; j++ )
                        center[j] *= scale;

                    if( iter > 0 )
                    {
                        double dist = 0;
                        const float* old_center = old_centers.ptr<float>(k);
                        for( j = 0; j < dims; j++ )
                        {
                            double t = center[j] - old_center[j];
                            dist += t*t;
                        }
                        max_center_shift = std::max(max_center_shift, dist);
                    }
                }
            }

            if( ++iter == MAX(criteria.maxCount, 2) || max_center_shift <= criteria.epsilon )
                break;

            // assign labels
            Mat dists(1, N, CV_64F);
            double* dist = dists.ptr<double>(0);
            parallel_for_(Range(0, N),
                         KMeansDistanceComputer(dist, labels, data, centers));
            compactness = 0;
            for( i = 0; i < N; i++ )
            {
                compactness += dist[i];
            }
        }

        if( compactness < best_compactness )
        {
            best_compactness = compactness;
            if( _centers.needed() )
                centers.copyTo(_centers);
            _labels.copyTo(best_labels);
        }
    }

    return best_compactness;
}