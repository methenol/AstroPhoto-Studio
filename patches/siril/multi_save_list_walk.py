"""Fix Siril's multi_save list walk in the AppImage binaries (siril, siril-cli); see README.md in this folder.

Siril 1.4.0 - 1.4.5, src/core/processing.c, multi_save():

    GList *list = multi_args->processed_images;
    while (list) {
        if (((struct _multi_split *)list->data)->index == out_index) { multi_data = list->data; break; }
        list = g_list_next(multi_args->processed_images);      /* upstream master: g_list_next(list) */
    }

compiles (x86-64, both binaries) to

    +144  48 8b 47 08     mov  0x8(%rdi),%rax     rax = head->next          <- patched to 48 8b 40 08: rax = rax->next
    +148  48 85 c0        test %rax,%rax
    +151  74 a3           je   (not found)
    +153  4c 8b 38        mov  (%rax),%r15        r15 = node->data
    +156  41 39 1f        cmp  %ebx,(%r15)        data->index == out_index ?
    +159  75 ef           jne  +144

The patch changes one ModRM byte (0x47 -> 0x40): the loop then advances from the current node, as upstream's
fix does.  The 17-byte loop must occur exactly once in each binary, or the build fails (a different Siril build
is not patched blindly).

usage: python multi_save_list_walk.py /opt/siril/usr/bin/siril-cli [/opt/siril/usr/bin/siril ...]
"""
import sys

BUGGY = bytes.fromhex("488b4708" "4885c0" "74a3" "4c8b38" "41391f" "75ef")
FIXED = bytes.fromhex("488b4008") + BUGGY[4:]


def patch(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    n_bug, n_fix = data.count(BUGGY), data.count(FIXED)
    if n_bug == 0 and n_fix == 1:
        return "already patched"
    if n_bug != 1 or n_fix != 0:
        raise SystemExit(f"{path}: multi_save loop found {n_bug}x (patched {n_fix}x), expected exactly 1 - "
                         "the Siril binary changed; re-check patches/siril/README.md")
    at = data.index(BUGGY)
    with open(path, "r+b") as f:
        f.seek(at)
        f.write(FIXED)
    return f"patched at 0x{at:x}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        print(f"{p}: {patch(p)}")
