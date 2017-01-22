DAFTAR LIBRARY OPENCV YANG BELUM KUPAHAMI

#include "precomp.hpp"

1.CV_INSTRUMENT_REGION()

2.Mat data= = _data.getMat(); 
APA ITU Mat::getMat() ?

3.int* labels = _labels.ptr<int>();
- Pointer labels menunjuk ke int 
- _labels bertipe Mat
- Apa itu Mat::ptr<>() , dan kenapa <int> ?

4.std::vector<int> counters(K); 
- Apakah ini maksudnya kapasitas vector = K ?

5.std::vector<Vec2f> _box(dims); 
- Apa itu tipe Vec2f? Bagaimana kerjanya?
- Liat librarynya : typedef Vec<float, 2> Vec2f;
template<typename _Tp, int n> class Vec : public Matx<_Tp, n, 1> {...};
- APA MAKSUD DARI INI?!
- Belajar template!!

6. Apa itu class Termcriteria ?
- Apa itu Termcriteria::type? 
- Apa itu Termcriteria::COUNT? 
- Apa itu Termcriteria::maxCount?
- Apa itu Termcriteria::epsilon?
- Apa itu Termcriteria::type?
- Apa itu FLT_EPSILON ?

7.const float* sample = data.ptr<float>(0);
- Apa ini?!
- pointer sample menunjuk konstanta dengan tipe data float ke
- data.ptr<float>(0)
- Apa maksud dari <float>?! Apa maksud dari (0)?!

8. Berikut baris :
    for( j = 0; j < dims; j++ )
        box[j] = Vec2f(sample[j], sample[j]);
- Apa itu Vec2f(sample[j],sample[j]);
- Apa yang dilakukan Vec2f?
- Kenapa dimasukin ke box yang berupa pointer menunjuk
  ke var _box dengan tipe Vec2f?
