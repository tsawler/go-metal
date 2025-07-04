// Main bridge implementation - modular
#import "bridge.h"

// Temporarily include remaining functions from old implementation
// We need to exclude the typedefs that are already in bridge_types.h

// Skip the typedef definitions that conflict with bridge_types.h
#define SKIP_BRIDGE_TYPEDEFS
#include "bridge_old.m.inc"