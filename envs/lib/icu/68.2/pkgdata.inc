GENCCODE_ASSEMBLY_TYPE=-a gcc-darwin
SO=dylib
SOBJ=dylib
A=a
LIBPREFIX=lib
LIB_EXT_ORDER=.68.2.dylib
COMPILE=arm64-apple-darwin20.0.0-clang -DU_ATTRIBUTE_DEPRECATED= -D_FORTIFY_SOURCE=2 -isystem /Users/shubhangimore/Text Segmentation/envs/include -mmacosx-version-min=11.0   -DU_HAVE_STRTOD_L=1 -DU_HAVE_XLOCALE_H=1 -DU_HAVE_STRING_VIEW=1  -ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe -isystem /Users/shubhangimore/Text Segmentation/envs/include -fdebug-prefix-map=/Users/runner/miniforge3/conda-bld/icu_1635409209597/work=/usr/local/src/conda/icu-68.2 -fdebug-prefix-map=/Users/shubhangimore/Text Segmentation/envs=/usr/local/src/conda-prefix -std=c11 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -fno-common -c
LIBFLAGS=-I/Users/shubhangimore/Text Segmentation/envs/include  -dynamic
GENLIB=arm64-apple-darwin20.0.0-clang -dynamiclib -dynamic -ftree-vectorize -fPIC -fPIE -fstack-protector-strong -O2 -pipe -isystem /Users/shubhangimore/Text Segmentation/envs/include -fdebug-prefix-map=/Users/runner/miniforge3/conda-bld/icu_1635409209597/work=/usr/local/src/conda/icu-68.2 -fdebug-prefix-map=/Users/shubhangimore/Text Segmentation/envs=/usr/local/src/conda-prefix -std=c11 -Wall -pedantic -Wshadow -Wpointer-arith -Wmissing-prototypes -Wwrite-strings   -Wl,-pie -Wl,-headerpad_max_install_names -Wl,-dead_strip_dylibs -Wl,-rpath,/Users/shubhangimore/Text Segmentation/envs/lib -L/Users/shubhangimore/Text Segmentation/envs/lib -Wl,-rpath,/Users/shubhangimore/Text Segmentation/envs/lib 
LDICUDTFLAGS=
LD_SONAME=-Wl,-compatibility_version -Wl,68 -Wl,-current_version -Wl,68.2 -install_name /Users/shubhangimore/Text Segmentation/envs/lib/
RPATH_FLAGS=
BIR_LDFLAGS=
AR=arm64-apple-darwin20.0.0-ar
ARFLAGS=r -c
RANLIB=arm64-apple-darwin20.0.0-ranlib
INSTALL_CMD=/usr/bin/install -c
