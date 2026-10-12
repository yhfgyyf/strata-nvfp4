#pragma once

// NVTX is header-only in the installed CUDA toolkit. Other backends keep diagnostics usable without it.
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define STRATA_PREFILL_LEDGER_NVTX 1
#endif

namespace strata::prefill {
struct LedgerScope {
    bool enabled;
    LedgerScope(bool on, const char* label) : enabled(on) {
#if defined(STRATA_PREFILL_LEDGER_NVTX)
        if (enabled) nvtxRangePushA(label);
#else
        (void) label;
#endif
    }
    ~LedgerScope() {
#if defined(STRATA_PREFILL_LEDGER_NVTX)
        if (enabled) nvtxRangePop();
#endif
    }
    LedgerScope(const LedgerScope&) = delete;
    LedgerScope& operator=(const LedgerScope&) = delete;
};
}
