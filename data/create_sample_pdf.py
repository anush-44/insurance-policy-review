from pathlib import Path

import pymupdf


def generate_sample_policy_pdf(output_path: Path | str = "data/sample_policy.pdf") -> Path:
    """Generate a realistic sample Indian retail health insurance policy PDF.

    Includes structured section numbering, numbered clauses, tables, and exclusions.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open()

    # --- Page 1: Preamble & Section 1: Scope of Cover ---
    page1 = doc.new_page()
    p1_text = """CARE HEALTH INSURANCE LIMITED
POLICY WORDING - CARE ADVANTAGE PLAN
UIN: CHIHLIP21183V032021

PREAMBLE
This Policy is a contract of insurance between Care Health Insurance Limited and the Insured Person named in the Policy Schedule.

SECTION 1: SCOPE OF COVERAGE

1.1 In-Patient Hospitalization
If an Insured Person is hospitalized during the Policy Period for Medically Necessary Treatment of an Illness or Injury, We will cover Medically Necessary Medical Expenses incurred up to the Sum Insured, provided hospitalization exceeds a minimum period of 24 consecutive hours.

1.2 Day Care Treatment
We will cover Medical Expenses incurred on Day Care Treatment requiring less than 24 hours hospitalization due to technological advancement, as listed in Annexure I.

1.3 Pre-Hospitalization and Post-Hospitalization Expenses
Medical expenses incurred 30 days prior to hospitalization and 60 days immediately post discharge from the hospital shall be payable under this benefit.
"""
    page1.insert_textbox(pymupdf.Rect(50, 50, 550, 750), p1_text, fontsize=10)

    # --- Page 2: Section 2: Limits & Co-Payment ---
    page2 = doc.new_page()
    p2_text = """SECTION 2: LIMITS, SUB-LIMITS AND CO-PAYMENT

2.1 Room Rent and ICU Capping
Coverage for Room Rent is capped at 1% of Sum Insured per day for Normal Single Private Room. Intensive Care Unit (ICU) charges are capped at 2% of Sum Insured per day. If the Insured occupies a room category higher than eligibility, proportionate deduction shall apply to associated medical expenses as per Clause 6.4.

2.2 Co-Payment Clause
A mandatory Co-Payment of 20% shall apply to all admissible claim amounts if the age of the Insured Person at inception of the first Policy is 61 years or above. The Insured shall bear 20% of each claim and the Company liability will be limited to 80%.

2.3 Cataract Sub-Limit
Our liability for Cataract treatment shall be limited to Rs. 40,000 per eye, subject to maximum of Rs. 80,000 per policy year.
"""
    page2.insert_textbox(pymupdf.Rect(50, 50, 550, 750), p2_text, fontsize=10)

    # --- Page 3: Section 3: Waiting Periods with Table ---
    page3 = doc.new_page()
    p3_text = """SECTION 3: WAITING PERIODS

3.1 Initial Waiting Period (30 Days)
Expenses related to the treatment of any illness within 30 days from the first policy commencement date shall be excluded except claims arising from an Accidental Injury.

3.2 Pre-Existing Diseases (PED) Waiting Period
Expenses related to the treatment of a Pre-Existing Disease (PED) and its direct complications shall be excluded until 36 months of continuous coverage have elapsed since the inception of the first policy.

3.3 Specific Disease Waiting Period (24 Months)
Expenses related to the treatment of listed conditions shall be excluded until 24 months of continuous coverage have elapsed:
"""
    page3.insert_textbox(pymupdf.Rect(50, 40, 550, 280), p3_text, fontsize=9)

    # Draw table for 3.3
    table_rect = pymupdf.Rect(50, 290, 540, 450)
    page3.draw_rect(table_rect, color=(0, 0, 0), width=1)
    # Header row
    page3.draw_line(pymupdf.Point(50, 320), pymupdf.Point(540, 320), color=(0, 0, 0), width=1)
    # Vertical dividers
    page3.draw_line(pymupdf.Point(200, 290), pymupdf.Point(200, 450), color=(0, 0, 0), width=1)
    page3.draw_line(pymupdf.Point(320, 290), pymupdf.Point(320, 450), color=(0, 0, 0), width=1)
    # Row dividers
    page3.draw_line(pymupdf.Point(50, 350), pymupdf.Point(540, 350), color=(0, 0, 0), width=0.5)
    page3.draw_line(pymupdf.Point(50, 380), pymupdf.Point(540, 380), color=(0, 0, 0), width=0.5)
    page3.draw_line(pymupdf.Point(50, 410), pymupdf.Point(540, 410), color=(0, 0, 0), width=0.5)

    # Insert table text
    page3.insert_textbox(pymupdf.Rect(55, 295, 195, 315), "Condition / Procedure", fontsize=9)
    page3.insert_textbox(pymupdf.Rect(205, 295, 315, 315), "Waiting Period", fontsize=9)
    page3.insert_textbox(
        pymupdf.Rect(325, 295, 535, 315), "Specific Exclusions / Sub-limits", fontsize=9
    )

    page3.insert_textbox(
        pymupdf.Rect(55, 325, 195, 345), "Benign Prostatic Hypertrophy", fontsize=8
    )
    page3.insert_textbox(pymupdf.Rect(205, 325, 315, 345), "24 Months", fontsize=8)
    page3.insert_textbox(
        pymupdf.Rect(325, 325, 535, 345), "Covered after 2 years continuous cover", fontsize=8
    )

    page3.insert_textbox(pymupdf.Rect(55, 355, 195, 375), "Cataract Surgery", fontsize=8)
    page3.insert_textbox(pymupdf.Rect(205, 355, 315, 375), "24 Months", fontsize=8)
    page3.insert_textbox(
        pymupdf.Rect(325, 355, 535, 375), "Subject to sub-limit under Clause 2.3", fontsize=8
    )

    page3.insert_textbox(pymupdf.Rect(55, 385, 195, 405), "Joint Replacement / Knee", fontsize=8)
    page3.insert_textbox(pymupdf.Rect(205, 385, 315, 405), "24 Months", fontsize=8)
    page3.insert_textbox(
        pymupdf.Rect(325, 385, 535, 405), "Unless due to accidental injury", fontsize=8
    )

    page3.insert_textbox(pymupdf.Rect(55, 415, 195, 445), "Hernia and Hydrocele", fontsize=8)
    page3.insert_textbox(pymupdf.Rect(205, 415, 315, 445), "24 Months", fontsize=8)
    page3.insert_textbox(pymupdf.Rect(325, 415, 535, 445), "Covered after 24 months", fontsize=8)

    # --- Page 4: Section 4: Exclusions & Section 5: Claims ---
    page4 = doc.new_page()
    p4_text = """SECTION 4: PERMANENT EXCLUSIONS

4.1 Standard Exclusions
The Company shall not be liable to make any payment for any expenses incurred by any Insured Person towards treatment in respect of:
a) Investigation & Evaluation (Code-Excl04): Expenses related to admission primarily for diagnostic purposes.
b) Rest Cure, Rehabilitation and Respite Care (Code-Excl05).
c) Obesity / Weight Control (Code-Excl06): Expenses related to surgical treatment of obesity unless meeting specific IRDAI BMI criteria.
d) Cosmetic or Plastic Surgery (Code-Excl08): Expenses for cosmetic or reconstructive treatment.
e) Maternity Expenses: Hospitalization expenses arising from pregnancy, childbirth, caesarean section, or abortion.

SECTION 5: CLAIM PROCEDURE

5.1 Cashless Facility Notice
For Planned Hospitalization at a Network Hospital, notice and pre-authorization request must be submitted at least 48 hours prior to admission. For Emergency Hospitalization, notice must be submitted within 24 hours of admission.

5.2 Reimbursement Claim Submission
Complete claim documents including original bills and discharge summary must be submitted to the Company within 30 days from the date of discharge.
"""
    page4.insert_textbox(pymupdf.Rect(50, 50, 550, 750), p4_text, fontsize=10)

    doc.save(str(output_path))
    doc.close()
    return output_path


if __name__ == "__main__":
    path = generate_sample_policy_pdf()
    print(f"Sample policy generated at: {path}")
