{ pkgs ? import <nixpkgs> {} }:

with pkgs;

let
  virtualenvDir = "pythonenv";
  manylinuxLibPath = stdenv.lib.makeLibraryPath [(callPackage ./manylinux1.nix {}).package];
  # liblapackShared = pkgs.liblapack.override { shared = true; };
in
mkShell {
  buildInputs = [
    busybox
    git
    # pkgconfig
    # hdf5
    # libzip
    # libpng
    # freetype
    # gfortran
    # liblapackShared
    nodejs

    (python36.withPackages (pythonPkgs: with pythonPkgs; [
      virtualenvwrapper
    ]))
  ];

  # Fix wheel building and init virtualenv
  shellHook = ''
    unset SOURCE_DATE_EPOCH
    if [ ! -d "${virtualenvDir}" ]; then
      virtualenv ${virtualenvDir}
    fi
    echo "manylinux1_compatible = True" > ${virtualenvDir}/lib/python3.6/_manylinux.py
    source ${virtualenvDir}/bin/activate
    export LD_LIBRARY_PATH=${manylinuxLibPath}
    export TMPDIR=/tmp
  '';
}