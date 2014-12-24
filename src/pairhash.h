#ifndef _PAIRHASH_H_
#define _PAIRHASH_H_
  
#include <functional>

template <class T>
inline void hash_combine(std::size_t & seed, const T & v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/*
inline void hash_combine(std::size_t & seed, const uint32_t v)
{
    std::hash<uint32_t> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}*/

class pairhash
{
    public:
    
    size_t operator()(const std::pair<uint32_t, uint32_t> & v) const
    {
      size_t seed = 0;
      hash_combine(seed, v.first);
      hash_combine(seed, v.second);
      return seed;
    }
};

#endif
