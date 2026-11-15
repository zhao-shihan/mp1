// Compile-fail runner self-check.  The probe must fail compilation and print
// an `error:` diagnostic; cmake/CompileFail.cmake rejects a successful
// compilation or a failure without that diagnostic.
#error "mp1 compile-fail runner self-check"
