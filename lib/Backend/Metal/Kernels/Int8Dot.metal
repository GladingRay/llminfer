inline int4 multiplyI8x4(char4 a, char4 b) { return int4(a) * int4(b); }

inline int sumInt4(int4 v) { return v.x + v.y + v.z + v.w; }

inline int dotI8x4(char4 a, char4 b) { return sumInt4(multiplyI8x4(a, b)); }
