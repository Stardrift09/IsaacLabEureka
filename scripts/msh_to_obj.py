"""
Convert MuJoCo binary .msh files to .obj format.
No Isaac Sim needed — plain Python only.

MuJoCo .msh binary layout:
  4 ints:          nvert, nnormal, ntexcoord, nface
  nvert*3 floats:  vertex positions (xyz)
  nnormal*3 floats: normals (xyz)
  ntexcoord*2 floats: UV coordinates
  nface*3 ints:    triangle indices

Run with:
    python3 scripts/msh_to_obj.py <input.msh> [output.obj]
Or convert all msh files in a directory tree:
    python3 scripts/msh_to_obj.py --dir <directory>
"""

import os
import struct
import sys


def msh_to_obj(msh_path: str, obj_path: str) -> None:
    with open(msh_path, "rb") as f:
        data = f.read()

    offset = 0
    nvert, nnormal, ntexcoord, nface = struct.unpack_from("<4i", data, offset)
    offset += 16

    verts = struct.unpack_from(f"<{nvert * 3}f", data, offset)
    offset += nvert * 12

    normals = struct.unpack_from(f"<{nnormal * 3}f", data, offset)
    offset += nnormal * 12

    uvs = struct.unpack_from(f"<{ntexcoord * 2}f", data, offset)
    offset += ntexcoord * 8

    faces = struct.unpack_from(f"<{nface * 3}i", data, offset)

    with open(obj_path, "w") as f:
        for i in range(nvert):
            f.write(f"v {verts[i*3]:.6f} {verts[i*3+1]:.6f} {verts[i*3+2]:.6f}\n")
        for i in range(nnormal):
            f.write(f"vn {normals[i*3]:.6f} {normals[i*3+1]:.6f} {normals[i*3+2]:.6f}\n")
        for i in range(ntexcoord):
            f.write(f"vt {uvs[i*2]:.6f} {uvs[i*2+1]:.6f}\n")
        has_uv = ntexcoord > 0
        has_n = nnormal > 0
        for i in range(nface):
            a, b, c = faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1
            if has_uv and has_n:
                f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
            elif has_n:
                f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                f.write(f"f {a} {b} {c}\n")


def convert_dir(root: str) -> None:
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.endswith(".msh"):
                msh = os.path.join(dirpath, fname)
                obj = msh[:-4] + ".obj"
                msh_to_obj(msh, obj)
                print(f"  {msh} → {obj}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--dir" in args:
        idx = args.index("--dir")
        convert_dir(args[idx + 1])
    elif len(args) >= 1:
        inp = args[0]
        out = args[1] if len(args) >= 2 else inp[:-4] + ".obj"
        msh_to_obj(inp, out)
        print(f"  {inp} → {out}")
    else:
        print(__doc__)
