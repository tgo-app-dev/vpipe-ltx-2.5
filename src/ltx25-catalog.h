#ifndef VPIPE_LTX25_CATALOG_H
#define VPIPE_LTX25_CATALOG_H

#include "stages/model-catalog.h"

#include <vector>

namespace ltx25 {

// The LTX-2.5 downloadable-model entries this plugin contributes, in
// display order. Handed to VpipePluginContext::register_catalog_entries
// at load, after which they behave exactly like built-in entries -- the
// drill-down menu, `model-fetch`, `model-select` and the web-ui model
// browser all see them.
std::vector<vpipe::ModelCatalogEntry> catalog_entries();

}  // namespace ltx25

#endif
