// Negative probe: 120 members exceed the MP1_REFLECTION limit.  The diagnostic must be the
// single class-level member-count static_assert naming the limit.

#include "mp1/mp1.hpp"
#include "test/support/reflection_limit_members.inc"

MP1_REFLECTION(reflection_limit_members, MP1_REFL_120_MEMBERS)
