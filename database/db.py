# -*- coding: utf-8 -*-
"""
Lớp truy cập dữ liệu cho bảng LichSuNhanDien trên SQL Server.
Dùng pyodbc + gọi các stored procedure đã tạo trong sqlserver_schema.sql.
"""
import pyodbc
from decimal import Decimal

# ----- CẤU HÌNH KẾT NỐI -----
SERVER_NAME = "DESKTOP-JD04BPC"     # tên server SQL Server của bạn
DATABASE_NAME = "ANPR_System"
DRIVER_NAME = "ODBC Driver 17 for SQL Server"

CONNECTION_STRING = (
    f"DRIVER={{{DRIVER_NAME}}};"
    f"SERVER={SERVER_NAME};"
    f"DATABASE={DATABASE_NAME};"
    f"Trusted_Connection=yes;"
)


def _get_conn():
    return pyodbc.connect(CONNECTION_STRING)


def _clean_value(value):
    """SQL Server trả DECIMAL/NUMERIC về kiểu Decimal, không tự chuyển được sang
    JSON. Ép về int/float thường để jsonify() trong Flask không bị lỗi."""
    if isinstance(value, Decimal):
        as_float = float(value)
        return int(as_float) if as_float.is_integer() else as_float
    return value


def _row_to_dict(columns, row):
    return {col: _clean_value(val) for col, val in zip(columns, row)}


def init_db():
    """Bảng và stored procedure đã được tạo sẵn bằng sqlserver_schema.sql trong SSMS,
    nên hàm này chỉ dùng để kiểm tra kết nối khi khởi động app."""
    conn = _get_conn()
    conn.close()


def insert_lich_su(bien_so_text, do_tin_cay, duong_dan_anh_goc, duong_dan_anh_crop):
    """Gọi sp_ThemLichSuNhanDien. Trả về ID bản ghi vừa thêm (kiểu int)."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "EXEC sp_ThemLichSuNhanDien ?, ?, ?, ?",
        (bien_so_text, do_tin_cay, duong_dan_anh_goc, duong_dan_anh_crop),
    )
    row = cur.fetchone()
    new_id = int(row[0]) if row and row[0] is not None else None
    conn.commit()
    conn.close()
    return new_id


def select_lich_su(tu_ngay=None, den_ngay=None, bien_so_tim=None, limit=50):
    """
    Gọi sp_LayLichSuNhanDien. Trả về danh sách dict, đã ép kiểu an toàn cho JSON.

    limit: chỉ lấy `limit` bản ghi MỚI NHẤT (sắp theo ID giảm dần). Sau nhiều ngày
    test, bảng có thể có hàng trăm bản ghi — nếu trả về hết, trang Lịch sử phải tải
    cùng lúc hàng trăm ảnh, làm trang cực chậm. Sắp xếp lại bằng Python (không dựa
    vào thứ tự ORDER BY bên trong stored procedure) để chắc chắn luôn là bản ghi
    mới nhất, bất kể procedure trả về theo thứ tự nào.
    Truyền limit=None để lấy hết (ví dụ khi cần xuất báo cáo đầy đủ).
    """
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "EXEC sp_LayLichSuNhanDien @TuNgay=?, @DenNgay=?, @BienSoTim=?",
        (tu_ngay, den_ngay, bien_so_tim),
    )
    columns = [col[0] for col in cur.description]
    rows = [_row_to_dict(columns, row) for row in cur.fetchall()]
    conn.close()
    rows.sort(key=lambda r: r.get("ID", 0), reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return rows


def select_chi_tiet(record_id):
    """Gọi sp_LayChiTietTheoID. Trả về 1 dict hoặc None."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("EXEC sp_LayChiTietTheoID @ID=?", (record_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        return None
    columns = [col[0] for col in cur.description]
    result = _row_to_dict(columns, row)
    conn.close()
    return result