-- =========================================================
-- Giai đoạn 1: Thiết kế CSDL - Microsoft SQL Server
-- =========================================================
CREATE DATABASE ANPR_System;
GO
USE ANPR_System;
GO

CREATE TABLE LichSuNhanDien (
    ID              INT IDENTITY(1,1) PRIMARY KEY,
    BienSo_Text     NVARCHAR(20)      NULL,
    DoTinCay        DECIMAL(5,2)      NULL,           -- % độ tin cậy, ví dụ 92.50
    DuongDanAnhGoc  NVARCHAR(260)     NOT NULL,
    DuongDanAnhCrop NVARCHAR(260)     NULL,
    ThoiGianQuet    DATETIME2         NOT NULL DEFAULT SYSDATETIME()
);
GO

CREATE INDEX IX_LichSuNhanDien_BienSo   ON LichSuNhanDien (BienSo_Text);
CREATE INDEX IX_LichSuNhanDien_ThoiGian ON LichSuNhanDien (ThoiGianQuet);
GO

-- ---------------------------------------------------------
-- Stored procedure: Thêm mới một lần quét
-- ---------------------------------------------------------
CREATE OR ALTER PROCEDURE sp_ThemLichSuNhanDien
    @BienSo_Text     NVARCHAR(20),
    @DoTinCay        DECIMAL(5,2),
    @DuongDanAnhGoc  NVARCHAR(260),
    @DuongDanAnhCrop NVARCHAR(260)
AS
BEGIN
    SET NOCOUNT ON;
    INSERT INTO LichSuNhanDien (BienSo_Text, DoTinCay, DuongDanAnhGoc, DuongDanAnhCrop, ThoiGianQuet)
    VALUES (@BienSo_Text, @DoTinCay, @DuongDanAnhGoc, @DuongDanAnhCrop, SYSDATETIME());

    SELECT SCOPE_IDENTITY() AS ID_MoiThem;
END
GO

-- ---------------------------------------------------------
-- Stored procedure: Lấy danh sách lịch sử, lọc theo ngày và biển số
-- ---------------------------------------------------------
CREATE OR ALTER PROCEDURE sp_LayLichSuNhanDien
    @TuNgay     DATETIME2 = NULL,
    @DenNgay    DATETIME2 = NULL,
    @BienSoTim  NVARCHAR(20) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    SELECT ID, BienSo_Text, DoTinCay, DuongDanAnhGoc, DuongDanAnhCrop, ThoiGianQuet
    FROM LichSuNhanDien
    WHERE (@TuNgay IS NULL OR ThoiGianQuet >= @TuNgay)
      AND (@DenNgay IS NULL OR ThoiGianQuet <= @DenNgay)
      AND (@BienSoTim IS NULL OR BienSo_Text LIKE '%' + @BienSoTim + '%')
    ORDER BY ThoiGianQuet DESC;
END
GO

-- ---------------------------------------------------------
-- Stored procedure: Lấy chi tiết 1 bản ghi theo ID (màn hình xem lại)
-- ---------------------------------------------------------
CREATE OR ALTER PROCEDURE sp_LayChiTietTheoID
    @ID INT
AS
BEGIN
    SET NOCOUNT ON;
    SELECT ID, BienSo_Text, DoTinCay, DuongDanAnhGoc, DuongDanAnhCrop, ThoiGianQuet
    FROM LichSuNhanDien
    WHERE ID = @ID;
END
GO