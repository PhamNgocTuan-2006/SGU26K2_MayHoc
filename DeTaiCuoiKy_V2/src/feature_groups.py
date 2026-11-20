def assign_group(col):
    c = col.lower()
    if c in ("id.orig_p", "id.resp_p"):
        return "cong"
    if c.startswith("proto") or c.startswith("service"):
        return "proto_service"
    if "flag" in c:
        return "co_TCP"
    if "window" in c:
        return "cua_so_TCP"
    if "iat" in c:
        return "thoi_gian_giua_goi_IAT"
    if "payload" in c:
        return "payload"
    if "header" in c:
        return "kich_thuoc_header"
    if c.startswith("active") or c.startswith("idle"):
        return "active_idle"
    if any(s in c for s in ("pkts", "subflow", "bulk", "ratio", "duration")):
        return "so_goi_va_luong"
    return "khac"


def group_columns(columns):
    # Gom tên các cột đặc trưng theo từng nhóm để dễ phân tích
    groups = {}
    for col in columns:
        grp = assign_group(col)
        if grp not in groups:
            groups[grp] = []
        groups[grp].append(col)
    return groups