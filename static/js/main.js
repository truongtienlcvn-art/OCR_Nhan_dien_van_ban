// =============================================================
// main.js - dùng chung, tự phát hiện đang ở trang nào để chạy đúng logic
// =============================================================

document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("video")) initScanPage();
    if (document.getElementById("historyTableBody")) initHistoryPage();
});

// -------------------------------------------------------------
// TRANG QUÉT BIỂN SỐ
// -------------------------------------------------------------
function initScanPage() {
    const video = document.getElementById("video");
    const canvas = document.getElementById("captureCanvas");
    const btnCapture = document.getElementById("btnCapture");
    const fileInput = document.getElementById("fileInput");
    const cameraHint = document.getElementById("cameraHint");
    const cameraStatusDot = document.getElementById("cameraStatusDot");

    // Mở camera nếu trình duyệt hỗ trợ và người dùng cho phép
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
            .then((stream) => {
                video.srcObject = stream;
                cameraHint.classList.add("is-hidden");
                cameraStatusDot.classList.remove("dot--idle");
                cameraStatusDot.classList.add("dot--live");
                btnCapture.disabled = false;
            })
            .catch(() => {
                cameraHint.textContent = "Không truy cập được camera — vẫn có thể dùng Upload ảnh.";
            });
    } else {
        cameraHint.textContent = "Trình duyệt không hỗ trợ camera — hãy dùng Upload ảnh.";
    }

    btnCapture.addEventListener("click", () => {
        const w = video.videoWidth || 640;
        const h = video.videoHeight || 480;
        canvas.width = w;
        canvas.height = h;
        canvas.getContext("2d").drawImage(video, 0, 0, w, h);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
        submitScan({ image_base64: dataUrl });
    });

    fileInput.addEventListener("change", () => {
        if (!fileInput.files || !fileInput.files[0]) return;
        const formData = new FormData();
        formData.append("image_file", fileInput.files[0]);
        submitScan(formData);
        fileInput.value = "";
    });

    // ---- Kéo-thả ảnh vào khung camera để nhận diện ----
    const dropZone = document.querySelector(".camera-frame");

    ["dragenter", "dragover"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("camera-frame--dragover");
        });
    });

    ["dragleave", "drop"].forEach((evt) => {
        dropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("camera-frame--dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (!file || !file.type.startsWith("image/")) return;
        const formData = new FormData();
        formData.append("image_file", file);
        submitScan(formData);
    });
}

async function submitScan(payload) {
    setScanLoading(true);
    hideScanError();
    try {
        let response;
        if (payload instanceof FormData) {
            response = await fetch("/api/scan", { method: "POST", body: payload });
        } else {
            response = await fetch("/api/scan", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
        }

        const rawText = await response.text();
        let data = null;
        try {
            data = JSON.parse(rawText);
        } catch (parseErr) {
            showScanError(
                `Server trả về dữ liệu không hợp lệ (HTTP ${response.status}). ` +
                `Xem terminal VS Code để biết lỗi Python chi tiết.`
            );
            return;
        }

        if (!response.ok) {
            showScanError((data && data.loi) || `Lỗi HTTP ${response.status}, xem terminal VS Code.`);
            return;
        }

        if (!data) {
            showScanError("Server không trả về dữ liệu kết quả, xem terminal VS Code.");
            return;
        }

        renderScanResult(data);
    } catch (err) {
        showScanError("Không kết nối được máy chủ: " + err.message);
    } finally {
        setScanLoading(false);
    }
}

function renderScanResult(data) {
    const plateReadout = document.getElementById("plateReadout");
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceFill = document.getElementById("confidenceFill");
    const latencyBadge = document.getElementById("latencyBadge");
    const imgInput = document.getElementById("imgInput");
    const imgOutput = document.getElementById("imgOutput");
    const emptyInput = document.getElementById("emptyInput");
    const emptyOutput = document.getElementById("emptyOutput");

    if (data.bien_so) {
        // Biển 2 dòng (xe máy) có 1 dấu cách ở giữa do backend tự chèn khi
        // phát hiện chuyển dòng -> hiển thị thành 2 dòng thật trên giao diện
        const formatted = escapeHtml(data.bien_so).replace(/ /g, "<br>");
        plateReadout.innerHTML = `<span class="plate-readout__text">${formatted}</span>`;
    } else if (data.phat_hien_duoc) {
        plateReadout.innerHTML = `<span class="plate-readout__text plate-readout__text--empty">PHÁT HIỆN VÙNG BIỂN SỐ NHƯNG KHÔNG ĐỌC ĐƯỢC CHỮ</span>`;
    } else {
        plateReadout.innerHTML = `<span class="plate-readout__text plate-readout__text--empty">KHÔNG PHÁT HIỆN BIỂN SỐ TRONG ẢNH</span>`;
    }

    const conf = data.do_tin_cay || 0;
    confidenceValue.textContent = `${conf}%`;
    confidenceFill.style.width = `${Math.min(conf, 100)}%`;
    confidenceFill.style.background = confidenceColor(conf);

    latencyBadge.textContent = `${data.thoi_gian_xu_ly_ms} ms`;

    imgInput.src = data.anh_dau_vao;
    imgInput.style.display = "block";
    emptyInput.classList.add("is-hidden");

    imgOutput.src = data.anh_dau_ra;
    imgOutput.style.display = "block";
    emptyOutput.classList.add("is-hidden");
}

function confidenceColor(conf) {
    if (conf >= 75) return "var(--success)";
    if (conf >= 40) return "var(--warning)";
    return "var(--danger)";
}

function setScanLoading(isLoading) {
    const btnCapture = document.getElementById("btnCapture");
    if (btnCapture && !btnCapture.disabled) {
        btnCapture.textContent = isLoading ? "Đang xử lý…" : "Chụp từ camera";
        btnCapture.disabled = isLoading;
        if (!isLoading) btnCapture.disabled = false;
    }
}

function showScanError(message) {
    const box = document.getElementById("scanError");
    box.textContent = message;
    box.classList.remove("is-hidden");
}

function hideScanError() {
    document.getElementById("scanError").classList.add("is-hidden");
}

// -------------------------------------------------------------
// TRANG LỊCH SỬ
// -------------------------------------------------------------
function initHistoryPage() {
    const form = document.getElementById("filterForm");
    const btnReset = document.getElementById("btnResetFilter");
    const modal = document.getElementById("detailModal");
    const modalClose = document.getElementById("modalClose");

    loadHistory();

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        loadHistory();
    });

    btnReset.addEventListener("click", () => {
        document.getElementById("filterBienSo").value = "";
        document.getElementById("filterTuNgay").value = "";
        document.getElementById("filterDenNgay").value = "";
        loadHistory();
    });

    modalClose.addEventListener("click", () => modal.classList.add("is-hidden"));
    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.classList.add("is-hidden");
    });
}

async function loadHistory() {
    const bienSo = document.getElementById("filterBienSo").value.trim();
    const tuNgay = document.getElementById("filterTuNgay").value;
    const denNgay = document.getElementById("filterDenNgay").value;

    const params = new URLSearchParams();
    if (bienSo) params.set("bien_so", bienSo);
    if (tuNgay) params.set("tu_ngay", tuNgay);
    if (denNgay) params.set("den_ngay", denNgay);

    const tbody = document.getElementById("historyTableBody");
    tbody.innerHTML = `<tr><td colspan="6" class="table-empty">Đang tải dữ liệu…</td></tr>`;

    try {
        const res = await fetch(`/api/history?${params.toString()}`);
        const rows = await res.json();

        if (!rows.length) {
            tbody.innerHTML = `<tr><td colspan="6" class="table-empty">Chưa có lần quét nào phù hợp.</td></tr>`;
            return;
        }

        tbody.innerHTML = rows.map(rowToHtml).join("");

        tbody.querySelectorAll("[data-view-id]").forEach((btn) => {
            btn.addEventListener("click", () => openDetail(btn.getAttribute("data-view-id")));
        });
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="6" class="table-empty">Lỗi tải dữ liệu: ${escapeHtml(err.message)}</td></tr>`;
    }
}

function rowToHtml(row) {
    const conf = row.DoTinCay || 0;
    const confClass = conf >= 75 ? "conf-chip--high" : conf >= 40 ? "conf-chip--mid" : "conf-chip--low";
    const thumb = row.DuongDanAnhCrop || row.DuongDanAnhGoc;
    return `
        <tr>
            <td>${row.ID}</td>
            <td><img class="thumb" src="${thumb}" alt="thumb"></td>
            <td class="plate-cell">${row.BienSo_Text ? escapeHtml(row.BienSo_Text) : "—"}</td>
            <td><span class="conf-chip ${confClass}">${conf}%</span></td>
            <td>${escapeHtml(row.ThoiGianQuet)}</td>
            <td><button class="link-btn" data-view-id="${row.ID}">Xem chi tiết</button></td>
        </tr>
    `;
}

// Suy ra đường dẫn ảnh của 1 bước xử lý khác từ đường dẫn đã có, bằng cách thay tiền tố
// tên file (vd "goc_" -> "det_") — không cần API/CSDL riêng cho từng bước, vì pipeline.py
// lưu mọi ảnh của cùng 1 lượt quét với cùng hậu tố thời gian (timestamp).
function deriveStepUrl(url, oldPrefix, newPrefix) {
    if (!url) return null;
    const slash = url.lastIndexOf("/");
    const dir = url.slice(0, slash + 1);
    const file = url.slice(slash + 1);
    if (!file.startsWith(oldPrefix)) return null;
    return dir + newPrefix + file.slice(oldPrefix.length);
}

// Gán ảnh cho 1 ô trong dải "các bước xử lý"; nếu không có đường dẫn (bước đó không áp
// dụng được, ví dụ không phát hiện được biển số thì không có ảnh tách ký tự), hoặc ảnh lỗi
// không tải được (ảnh quét từ trước khi có tính năng này), ẩn hẳn ô đó thay vì hiện ảnh vỡ.
function setStepImage(itemId, imgId, url) {
    const item = document.getElementById(itemId);
    const img = document.getElementById(imgId);
    if (!url) {
        item.classList.add("is-hidden");
        return;
    }
    item.classList.remove("is-hidden");
    img.onerror = () => item.classList.add("is-hidden");
    img.src = url;
}

async function openDetail(id) {
    const modal = document.getElementById("detailModal");
    try {
        const res = await fetch(`/api/history/${id}`);
        const data = await res.json();
        if (!res.ok) return;

        document.getElementById("modalId").textContent = data.ID;
        setStepImage("stepItem-goc", "modalStepGoc", data.DuongDanAnhGoc);
        setStepImage("stepItem-det", "modalStepDet", deriveStepUrl(data.DuongDanAnhGoc, "goc_", "det_"));
        setStepImage("stepItem-crop", "modalStepCrop", data.DuongDanAnhCrop);
        setStepImage("stepItem-char", "modalStepChar", deriveStepUrl(data.DuongDanAnhCrop, "crop_", "char_"));
        setStepImage("stepItem-out", "modalStepOut", deriveStepUrl(data.DuongDanAnhGoc, "goc_", "out_"));
        document.getElementById("modalBienSo").textContent = data.BienSo_Text || "Không đọc được";
        document.getElementById("modalDoTinCay").textContent = `${data.DoTinCay || 0}%`;
        document.getElementById("modalThoiGian").textContent = data.ThoiGianQuet;

        modal.classList.remove("is-hidden");
    } catch (err) {
        alert("Không tải được chi tiết: " + err.message);
    }
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
// ---------------------------------------------------------------------------
// Chuyển đổi giao diện sáng / tối
// ---------------------------------------------------------------------------
(function initThemeToggle() {
    const root = document.documentElement;
    const btn = document.getElementById("themeToggle");
    if (!btn) return;
    const icon = btn.querySelector(".theme-toggle__icon");

    function isDark() {
        return root.getAttribute("data-theme") === "dark";
    }

    function syncIcon() {
        // Hiện icon mặt trăng khi đang sáng (gợi ý bấm để chuyển sang tối) và
        // ngược lại, theo đúng quy ước thường thấy của nút chuyển theme.
        icon.textContent = isDark() ? "\u2600\uFE0F" : "\uD83C\uDF19";
    }
    syncIcon();

    btn.addEventListener("click", () => {
        const nextDark = !isDark();
        if (nextDark) {
            root.setAttribute("data-theme", "dark");
        } else {
            root.removeAttribute("data-theme");
        }
        try {
            localStorage.setItem("anpr-theme", nextDark ? "dark" : "light");
        } catch (e) {
            // localStorage bị chặn (chế độ ẩn danh...) -> vẫn đổi giao diện được,
            // chỉ là không nhớ lựa chọn cho lần sau.
        }
        syncIcon();
    });
})();