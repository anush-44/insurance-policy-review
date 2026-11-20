"""Download and validate official health policy wording PDFs in policies.csv.

Run with ``python -m data.acquire_policies``. Existing verified files are reused.
The script never replaces a different existing file unless ``--refresh`` is set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
FIELDS = [
    "policy_id",
    "insurer",
    "product_name",
    "uin",
    "plan_type",
    "pages",
    "source_url",
    "download_date",
    "sha256",
    "local_path",
    "validation_status",
]
OFFICIAL_HOSTS: dict[str, tuple[str, ...]] = {
    "icici lombard": ("icicilombard.com",),
    "care": ("careinsurance.com",),
    "hdfc ergo": ("hdfcergo.com", "customer-portal-assets.hdfcergo.com"),
    "niva bupa": ("nivabupa.com", "transactions.nivabupa.com"),
    "aditya birla": ("adityabirlacapital.com", "adityabirlahealthinsurance.com"),
    "star health": ("starhealth.in", "d28c6jni2fmamz.cloudfront.net", "cdn.starhealth.in"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_url(row: dict[str, str]) -> bool:
    host = (urlparse(row["source_url"]).hostname or "").lower()
    insurer = row["insurer"].lower()
    candidates = next((domains for name, domains in OFFICIAL_HOSTS.items() if name in insurer), ())
    return urlparse(row["source_url"]).scheme == "https" and any(
        host == domain or host.endswith("." + domain) for domain in candidates
    )


def validate_pdf(path: Path, row: dict[str, str]) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                return {
                    "status": "invalid_pdf_signature",
                    "pages": None,
                    "text_chars": 0,
                    "uin_detected": None,
                }
        with pymupdf.open(path) as doc:
            pages = doc.page_count
            text = "\n".join(page.get_text("text") for page in doc)
    except Exception as exc:
        return {
            "status": "corrupt_or_unreadable_pdf",
            "error": str(exc),
            "pages": None,
            "text_chars": 0,
            "uin_detected": None,
        }

    flat_text = re.sub(r"\s+", " ", text).lower()
    generic_tokens = {"health", "insurance", "policy", "plan", "the", "and"}
    product_tokens = [
        token.lower()
        for token in re.findall(r"[a-z0-9]+", row["product_name"])
        if len(token) > 2 and token.lower() not in generic_tokens
    ]
    product_found = bool(product_tokens) and all(token in flat_text for token in product_tokens)
    detected_uins = sorted(set(re.findall(r"\b[A-Z]{2,8}HLIP\d{5}V\d{6}\b", text.upper())))
    expected_uin = row["uin"].strip().upper()
    try:
        expected_pages = int(row["pages"]) if row["pages"].strip() else None
    except ValueError:
        expected_pages = -1
    if expected_pages == -1:
        status = "invalid_manifest_page_count"
    elif pages < 1:
        status = "invalid_page_count"
    elif expected_pages is not None and expected_pages != pages:
        status = "page_count_mismatch"
    elif len(text.strip()) < 200:
        status = "scanned_or_insufficient_selectable_text"
    elif not product_found:
        status = "product_name_not_found"
    elif expected_uin and detected_uins and expected_uin not in detected_uins:
        status = "uin_mismatch"
    elif expected_uin and not detected_uins:
        status = "valid_uin_not_detected"
    else:
        status = "valid"
    return {
        "status": status,
        "pages": pages,
        "text_chars": len(text.strip()),
        "uin_detected": detected_uins,
        "product_found": product_found,
    }


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Manifest columns must be exactly: {', '.join(FIELDS)}")
        rows = list(reader)
    ids = [row["policy_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest policy_id values must be unique")
    return rows


def download(url: str, target: Path, retries: int, timeout: float) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "PolicyLens-policy-data/1.0"})
            with urlopen(request, timeout=timeout) as response, target.open("wb") as output:
                if not response.geturl().startswith("https://"):
                    raise ValueError("Redirected to a non-HTTPS URL")
                shutil.copyfileobj(response, output)
            return response.geturl()
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"download failed after {retries + 1} attempts: {last_error}")


def run(manifest: Path, refresh: bool, retries: int, timeout: float, report_path: Path) -> int:
    rows = read_manifest(manifest)
    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest),
        "results": [],
    }
    results: list[dict[str, object]] = []
    for row in rows:
        result: dict[str, object] = {"policy_id": row["policy_id"], "source_url": row["source_url"]}
        target = (ROOT / row["local_path"]).resolve()
        if ROOT.resolve() not in target.parents:
            result["status"] = "unsafe_local_path"
            results.append(result)
            continue
        if not row["source_url"] or not official_url(row):
            result["status"] = "missing_or_non_official_source_url"
            results.append(result)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and not refresh:
            digest = sha256_file(target)
            old_digest = row["sha256"].strip()
            checked = validate_pdf(target, row)
            if checked["status"] in {"valid", "valid_uin_not_detected"} and (
                not old_digest or digest == old_digest
            ):
                row["pages"] = str(checked["pages"])
                row["sha256"] = digest
                # A locally supplied file has no acquisition timestamp unless the
                # manifest already records one; content validation cannot prove provenance.
                row["validation_status"] = (
                    str(checked["status"])
                    if row["download_date"]
                    else "valid_pdf_source_unverified"
                )
                result.update(checked)
                result["status"] = (
                    "reused_validated_file"
                    if row["download_date"]
                    else "existing_local_file_source_unverified"
                )
                result["sha256"] = digest
                results.append(result)
                continue
            if old_digest and digest != old_digest:
                result.update(checked)
                result["status"] = "existing_file_hash_changed; not overwritten"
                results.append(result)
                continue

        temp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                prefix="policylens-", suffix=".part", delete=False, dir=target.parent
            ) as temp:
                temp_name = temp.name
            temp_path = Path(temp_name)
            final_url = download(row["source_url"], temp_path, retries, timeout)
            final_host = (urlparse(final_url).hostname or "").lower()
            allowed = next(
                (
                    domains
                    for name, domains in OFFICIAL_HOSTS.items()
                    if name in row["insurer"].lower()
                ),
                (),
            )
            if not any(
                final_host == domain or final_host.endswith("." + domain) for domain in allowed
            ):
                raise ValueError(f"Redirect ended on unexpected host: {final_host}")
            checked = validate_pdf(temp_path, row)
            result.update(checked)
            if checked["status"] not in {"valid", "valid_uin_not_detected"}:
                result["status"] = checked["status"]
                results.append(result)
                continue
            new_hash = sha256_file(temp_path)
            if target.exists() and sha256_file(target) != new_hash and not refresh:
                result["status"] = "different_version_exists; not overwritten"
                result["sha256"] = new_hash
                results.append(result)
                continue
            temp_path.replace(target)
            row["pages"] = str(checked["pages"])
            row["download_date"] = date.today().isoformat()
            row["sha256"] = new_hash
            row["validation_status"] = str(checked["status"])
            result.update(
                {"status": "downloaded_and_validated", "final_url": final_url, "sha256": new_hash}
            )
        except Exception as exc:
            result["status"] = "download_or_validation_error"
            result["error"] = str(exc)
        finally:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)
        results.append(result)

    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report["results"] = results
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in results:
        print(f"{item['policy_id']}: {item['status']}")
    good = sum(
        item["status"] in {"downloaded_and_validated", "reused_validated_file"} for item in results
    )
    local_only = sum(item["status"] == "existing_local_file_source_unverified" for item in results)
    print(
        f"Officially acquired and validated: {good}/{len(rows)}; local files with unverified provenance: {local_only}; report: {report_path}"
    )
    return 0 if good else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/policies.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "data/acquisition_report.json")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Allow replacing existing files with newly validated versions",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    sys.exit(run(args.manifest, args.refresh, args.retries, args.timeout, args.report))


if __name__ == "__main__":
    main()
