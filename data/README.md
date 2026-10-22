# PolicyLens Dataset & Policy Sources

## Overview
PolicyLens is designed specifically for Indian Retail Health Insurance policy wording documents (usually 30–60 pages PDFs published by IRDAI registered insurers).

## Typical Sample Policies
1. **Care Health Insurance (Care Plan)**: Standard retail indemnity plan with waiting period tables, sub-limits, and co-payment clauses.
2. **HDFC ERGO Optima Secure**: Features 2X coverage benefit, zero room rent capping, standard 36-month/24-month waiting periods.
3. **Star Health Comprehensive**: Features specific cataract and joint replacement sub-limits, tiered room rent options.
4. **Niva Bupa ReAssure 2.0**: Age-lock features, cashless claim process clauses, modern day-care treatment lists.

## Sample PDF Generator & Test Policies
For CI/CD and self-contained testing, `data/create_sample_pdf.py` synthesizes a representative Indian health insurance policy document containing:
- Clause 1.1: Scope of Cover (In-patient Hospitalization)
- Clause 2.1: Room Rent Capping and ICU Sub-limits
- Clause 3.1: Co-payment terms (20% co-pay on senior age tiers)
- Clause 4.1: Initial Waiting Period (30 days)
- Clause 4.2: Waiting Period for Pre-Existing Diseases (PED - 36 months)
- Clause 4.3: Specific Disease Waiting Period Table (24 months for Cataract, Hernia, Joint Replacement)
- Clause 5.1: Permanent Exclusions (Cosmetic surgery, Adventure sports, Domiciliary maternity)
- Clause 6.1: Cashless Claim Notice Procedure
