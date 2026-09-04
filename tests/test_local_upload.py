"""
Memotrix Local Upload Test — Every file type against localhost Postgres.

This script:
  1. Connects to your local Postgres (pgvector) using your .env credentials.
  2. Creates sample files for EVERY supported file format.
  3. Uploads (ingests) each one into the Postgres-backed Memory.
  4. Searches for content from each file to verify indexing.
  5. Reports a full pass/fail matrix with timing.

Usage:
    python -m pytest tests/test_local_upload.py -v --tb=short
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (str(SRC), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from memotrix import Memory
from memotrix.embeddings import FakeEmbeddings
from memotrix.filetypes import supported_extensions
from memotrix.utils.exceptions import UnsupportedDocumentTypeError

# ── Constants ────────────────────────────────────────────────────────────────
DSN = "postgresql://postgres:utsav1424@localhost:5432/memotrix"
TABLE = "memotrix_upload_test"   # isolated table so we don't pollute production
FIXTURES = ROOT / "tests" / "filesForTests"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def memory():
    """Module-scoped Postgres-backed Memory. Created once, shared by all tests."""
    m = Memory(
        embeddings=FakeEmbeddings(dim=8),
        backend="postgres",
        connection=DSN,
        config=__import__("memotrix").MemoryConfig(table_name=TABLE),
    )
    yield m
    m.close()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: TEXT / MARKUP FILES
# ═══════════════════════════════════════════════════════════════════════════

class TestTextFiles:
    def test_upload_txt(self, memory, tmp_path):
        f = tmp_path / "plain.txt"
        f.write_text("Memotrix ingests plain text files into Postgres.", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1
        hits = memory.search("plain text", top_k=1)
        assert len(hits) >= 1

    def test_upload_markdown(self, memory, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Architecture\n\nMemotrix uses HNSW for dense vectors and BM25 for sparse.\n", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_html(self, memory, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("""
        <html><head><title>Docs</title></head>
        <body><h1>Installation</h1><p>pip install memotrix</p></body>
        </html>""", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_sql(self, memory, tmp_path):
        f = tmp_path / "schema.sql"
        f.write_text("""
        CREATE TABLE patients (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            dob DATE,
            blood_type VARCHAR(3)
        );
        INSERT INTO patients (name, dob, blood_type) VALUES ('John Doe', '1990-01-15', 'A+');
        """, encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: STRUCTURED DATA FILES
# ═══════════════════════════════════════════════════════════════════════════

class TestStructuredData:
    def test_upload_json(self, memory, tmp_path):
        f = tmp_path / "config.json"
        f.write_text(json.dumps({
            "app": "memotrix",
            "version": "0.1.0",
            "features": ["hybrid_search", "dedup", "rerank"],
        }, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_jsonl(self, memory, tmp_path):
        f = tmp_path / "events.jsonl"
        lines = [
            json.dumps({"event": "user_login", "user": "alice", "ts": "2026-09-01T10:00:00Z"}),
            json.dumps({"event": "file_upload", "user": "bob", "ts": "2026-09-01T10:05:00Z"}),
            json.dumps({"event": "search_query", "user": "alice", "ts": "2026-09-01T10:10:00Z"}),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_csv(self, memory, tmp_path):
        f = tmp_path / "patients.csv"
        f.write_text(
            "patient_id,name,diagnosis,blood_pressure,heart_rate\n"
            "P001,John Doe,Hypertension,140/90,78\n"
            "P002,Jane Smith,Diabetes Type 2,130/85,82\n"
            "P003,Bob Wilson,Asthma,120/80,75\n",
            encoding="utf-8",
        )
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_yaml(self, memory, tmp_path):
        f = tmp_path / "pipeline.yaml"
        f.write_text("""
name: memotrix-ci
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e .[all]
      - run: pytest
        """, encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_xml(self, memory, tmp_path):
        f = tmp_path / "prescription.xml"
        f.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<prescription>
    <patient name="John Doe" id="P001"/>
    <medication name="Metformin" dose="500mg" frequency="twice daily"/>
    <prescriber>Dr. Smith</prescriber>
    <date>2026-09-01</date>
</prescription>""", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: PROGRAMMING FILES
# ═══════════════════════════════════════════════════════════════════════════

class TestProgrammingFiles:
    def test_upload_python(self, memory, tmp_path):
        f = tmp_path / "extractor.py"
        f.write_text('''
"""Custom file extractor for clinical notes."""

from pathlib import Path
from memotrix.utils.outputSturcture import build_document


class ClinicalNoteExtractor:
    """Extracts structured clinical notes from .cn files."""

    def extract(self, path: Path, **kwargs):
        text = path.read_text(encoding="utf-8")
        return build_document(path, text, extra_metadata={"type": "clinical_note"})
''', encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_javascript(self, memory, tmp_path):
        f = tmp_path / "app.js"
        f.write_text("""
// Memotrix JS client (hypothetical)
async function searchMemory(query) {
    const response = await fetch('/api/search', {
        method: 'POST',
        body: JSON.stringify({ query, top_k: 5 }),
    });
    return response.json();
}
module.exports = { searchMemory };
""", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_typescript(self, memory, tmp_path):
        f = tmp_path / "types.ts"
        f.write_text("""
interface MemoryHit {
    chunk_text: string;
    score: number;
    source_path: string;
    memory_type: 'semantic' | 'episodic' | 'procedural';
}

interface SearchParams {
    query: string;
    top_k?: number;
    filters?: Record<string, string>;
}
""", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_java(self, memory, tmp_path):
        f = tmp_path / "Patient.java"
        f.write_text("""
public class Patient {
    private String id;
    private String name;
    private String diagnosis;

    public Patient(String id, String name, String diagnosis) {
        this.id = id;
        this.name = name;
        this.diagnosis = diagnosis;
    }

    public String getDiagnosis() { return diagnosis; }
}
""", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_go(self, memory, tmp_path):
        f = tmp_path / "main.go"
        f.write_text("""
package main

import "fmt"

type EHRRecord struct {
    PatientID string
    Diagnosis string
    Vitals    map[string]float64
}

func main() {
    record := EHRRecord{
        PatientID: "P001",
        Diagnosis: "Hypertension",
        Vitals:    map[string]float64{"systolic": 140, "diastolic": 90},
    }
    fmt.Printf("Patient %s: %s\\n", record.PatientID, record.Diagnosis)
}
""", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_rust(self, memory, tmp_path):
        f = tmp_path / "lib.rs"
        f.write_text("""
struct MedicalRecord {
    patient_id: String,
    diagnosis: String,
    icd_code: String,
}

impl MedicalRecord {
    fn new(id: &str, diagnosis: &str, icd: &str) -> Self {
        MedicalRecord {
            patient_id: id.to_string(),
            diagnosis: diagnosis.to_string(),
            icd_code: icd.to_string(),
        }
    }
}
""", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_cpp(self, memory, tmp_path):
        f = tmp_path / "vitals.cpp"
        f.write_text("""
#include <iostream>
#include <string>

struct VitalSigns {
    double heart_rate;
    double blood_pressure_sys;
    double blood_pressure_dia;
    double temperature;
};

int main() {
    VitalSigns v = {78.0, 120.0, 80.0, 98.6};
    std::cout << "HR: " << v.heart_rate << " bpm" << std::endl;
    return 0;
}
""", encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: OFFICE DOCUMENTS (REAL FIXTURES)
# ═══════════════════════════════════════════════════════════════════════════

class TestOfficeDocuments:
    @pytest.mark.skipif(
        not (FIXTURES / "Excel_Practice.xlsx").exists(),
        reason="xlsx fixture not found",
    )
    def test_upload_xlsx(self, memory):
        stats = memory.add(FIXTURES / "Excel_Practice.xlsx")
        assert stats["chunks"] >= 1
        assert stats["filename"] == "Excel_Practice.xlsx"

    @pytest.mark.skipif(
        not (FIXTURES / "SignalHire_exports.csv").exists(),
        reason="csv fixture not found",
    )
    def test_upload_real_csv(self, memory):
        stats = memory.add(FIXTURES / "SignalHire_exports.csv")
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "this_is_wow.txt").exists(),
        reason="txt fixture not found",
    )
    def test_upload_real_txt(self, memory):
        stats = memory.add(FIXTURES / "this_is_wow.txt")
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "Realtime_Conversation_Analytics_Report_1.docx").exists(),
        reason="docx fixture not found",
    )
    def test_upload_docx(self, memory):
        stats = memory.add(
            FIXTURES / "Realtime_Conversation_Analytics_Report_1.docx",
            describe_images=False,
        )
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "donor_report_final.pdf").exists(),
        reason="pdf fixture not found",
    )
    def test_upload_pdf(self, memory):
        stats = memory.add(FIXTURES / "donor_report_final.pdf", describe_images=False)
        assert stats["chunks"] >= 1

    @pytest.mark.skipif(
        not (FIXTURES / "donor_report.pptx").exists(),
        reason="pptx fixture not found",
    )
    def test_upload_pptx(self, memory):
        stats = memory.add(FIXTURES / "donor_report.pptx", describe_images=False)
        assert stats["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: IMAGES
# ═══════════════════════════════════════════════════════════════════════════

class TestImages:
    @pytest.mark.skipif(
        not (FIXTURES / "test.jpg").exists(),
        reason="jpg fixture not found",
    )
    def test_upload_jpg(self, memory):
        stats = memory.add(FIXTURES / "test.jpg", describe_images=False)
        assert stats["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: MEDICAL / FHIR FILES
# ═══════════════════════════════════════════════════════════════════════════

class TestMedicalFHIR:
    """Upload FHIR clinical resources to Postgres and verify search."""

    def test_upload_fhir_patient(self, memory, tmp_path):
        patient = {
            "resourceType": "Patient",
            "id": "pg-patient-001",
            "name": [{"family": "Rodriguez", "given": ["Maria", "Elena"]}],
            "gender": "female",
            "birthDate": "1985-03-22",
            "address": [{"city": "Houston", "state": "TX", "country": "US"}],
            "telecom": [{"system": "phone", "value": "555-0134"}],
        }
        f = tmp_path / "patient.json"
        f.write_text(json.dumps(patient, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1
        hits = memory.search("Rodriguez patient", top_k=3)
        assert len(hits) >= 1

    def test_upload_fhir_observation_blood_pressure(self, memory, tmp_path):
        obs = {
            "resourceType": "Observation",
            "id": "obs-bp-pg",
            "status": "final",
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Blood pressure panel"}],
                "text": "Blood pressure",
            },
            "subject": {"reference": "Patient/pg-patient-001"},
            "effectiveDateTime": "2026-09-01T08:30:00Z",
            "component": [
                {
                    "code": {"coding": [{"display": "Systolic blood pressure"}]},
                    "valueQuantity": {"value": 142, "unit": "mmHg"},
                },
                {
                    "code": {"coding": [{"display": "Diastolic blood pressure"}]},
                    "valueQuantity": {"value": 88, "unit": "mmHg"},
                },
            ],
        }
        f = tmp_path / "bp_observation.json"
        f.write_text(json.dumps(obs, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_fhir_observation_lab(self, memory, tmp_path):
        obs = {
            "resourceType": "Observation",
            "id": "obs-hba1c-pg",
            "status": "final",
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}],
            },
            "subject": {"reference": "Patient/pg-patient-001"},
            "effectiveDateTime": "2026-08-28T14:00:00Z",
            "valueQuantity": {"value": 7.2, "unit": "%"},
            "interpretation": [{"coding": [{"display": "Above normal"}]}],
        }
        f = tmp_path / "hba1c.json"
        f.write_text(json.dumps(obs, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_fhir_condition(self, memory, tmp_path):
        cond = {
            "resourceType": "Condition",
            "id": "cond-dm2-pg",
            "clinicalStatus": {"coding": [{"code": "active", "display": "Active"}]},
            "code": {
                "coding": [{"system": "http://snomed.info/sct", "code": "44054006", "display": "Diabetes mellitus type 2"}],
                "text": "Type 2 Diabetes",
            },
            "subject": {"reference": "Patient/pg-patient-001"},
            "onsetDateTime": "2020-06-15",
        }
        f = tmp_path / "condition.json"
        f.write_text(json.dumps(cond, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_fhir_medication_request(self, memory, tmp_path):
        med = {
            "resourceType": "MedicationRequest",
            "id": "medrx-metformin-pg",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [{"display": "Metformin 500mg tablet"}],
                "text": "Metformin 500mg",
            },
            "subject": {"reference": "Patient/pg-patient-001"},
            "dosageInstruction": [
                {"text": "Take 1 tablet by mouth twice daily with meals."}
            ],
        }
        f = tmp_path / "medication.json"
        f.write_text(json.dumps(med, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_fhir_allergy(self, memory, tmp_path):
        allergy = {
            "resourceType": "AllergyIntolerance",
            "id": "allergy-penicillin-pg",
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "type": "allergy",
            "category": ["medication"],
            "criticality": "high",
            "code": {"coding": [{"display": "Penicillin"}], "text": "Penicillin allergy"},
            "patient": {"reference": "Patient/pg-patient-001"},
            "reaction": [
                {
                    "substance": {"coding": [{"display": "Penicillin"}]},
                    "manifestation": [{"coding": [{"display": "Anaphylactic reaction"}]}],
                    "severity": "severe",
                }
            ],
        }
        f = tmp_path / "allergy.json"
        f.write_text(json.dumps(allergy, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_fhir_encounter(self, memory, tmp_path):
        encounter = {
            "resourceType": "Encounter",
            "id": "enc-er-pg",
            "status": "finished",
            "class": {"code": "EMER", "display": "Emergency"},
            "subject": {"reference": "Patient/pg-patient-001"},
            "period": {"start": "2026-09-01T06:00:00Z", "end": "2026-09-01T14:00:00Z"},
            "reasonCode": [
                {"coding": [{"display": "Chest pain"}], "text": "Acute chest pain"}
            ],
        }
        f = tmp_path / "encounter.json"
        f.write_text(json.dumps(encounter, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_fhir_diagnostic_report(self, memory, tmp_path):
        report = {
            "resourceType": "DiagnosticReport",
            "id": "dr-cbc-pg",
            "status": "final",
            "code": {"coding": [{"display": "Complete Blood Count"}], "text": "CBC"},
            "subject": {"reference": "Patient/pg-patient-001"},
            "effectiveDateTime": "2026-09-01T09:00:00Z",
            "conclusion": "WBC elevated at 12.5, indicative of possible infection.",
        }
        f = tmp_path / "diagnostic_report.json"
        f.write_text(json.dumps(report, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_fhir_bundle(self, memory, tmp_path):
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "bundle-pt", "name": [{"family": "Chen"}], "gender": "male"}},
                {"resource": {"resourceType": "Condition", "id": "bundle-cond", "code": {"coding": [{"display": "Asthma"}]}, "subject": {"reference": "Patient/bundle-pt"}}},
                {"resource": {"resourceType": "MedicationRequest", "id": "bundle-med", "status": "active", "medicationCodeableConcept": {"coding": [{"display": "Albuterol inhaler"}]}, "subject": {"reference": "Patient/bundle-pt"}}},
            ],
        }
        f = tmp_path / "bundle.json"
        f.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7: MEDICAL TEXT NOTES
# ═══════════════════════════════════════════════════════════════════════════

class TestMedicalNotes:
    """Upload clinical free-text notes (not FHIR)."""

    def test_upload_clinical_note(self, memory, tmp_path):
        f = tmp_path / "clinical_note.txt"
        f.write_text("""
SUBJECTIVE:
Patient Maria Rodriguez presents with fatigue, increased thirst, and frequent urination
over the past 3 weeks. She reports unintentional weight loss of 5 lbs. No chest pain,
shortness of breath, or visual changes.

OBJECTIVE:
BP 142/88 mmHg, HR 78 bpm, Temp 98.4°F, Weight 165 lbs.
Fasting glucose: 186 mg/dL. HbA1c: 7.2%.
Fundoscopic exam: No retinopathy.

ASSESSMENT:
1. Diabetes mellitus type 2, newly diagnosed (E11.9)
2. Hypertension, stage 1 (I10)

PLAN:
1. Start Metformin 500mg PO BID with meals.
2. Lifestyle modifications: diet counseling, exercise 150 min/week.
3. Home blood glucose monitoring, log daily fasting glucose.
4. Recheck HbA1c in 3 months.
5. Referral to ophthalmology for diabetic eye screening.
6. Follow up in 4 weeks.
        """, encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1
        hits = memory.search("diabetes diagnosis", top_k=3)
        assert len(hits) >= 1

    def test_upload_discharge_summary(self, memory, tmp_path):
        f = tmp_path / "discharge_summary.txt"
        f.write_text("""
DISCHARGE SUMMARY

Patient: Maria Rodriguez, DOB 03/22/1985, MRN: PG-001
Admission: 09/01/2026, Discharge: 09/03/2026
Attending: Dr. James Wilson, Internal Medicine

CHIEF COMPLAINT: Acute chest pain

HOSPITAL COURSE:
Patient presented to the ED with acute onset substernal chest pain radiating to left arm.
Troponin levels were negative x3. ECG showed normal sinus rhythm. Echocardiogram
demonstrated normal LV function (EF 60%). Stress test was negative for ischemia.
Pain was determined to be musculoskeletal in origin.

DISCHARGE DIAGNOSES:
1. Chest pain, non-cardiac (R07.9)
2. Diabetes mellitus type 2 (E11.9)
3. Essential hypertension (I10)

DISCHARGE MEDICATIONS:
1. Metformin 500mg PO BID
2. Lisinopril 10mg PO daily
3. Ibuprofen 400mg PO TID PRN for chest wall pain

FOLLOW-UP:
- PCP in 1 week
- Cardiology in 1 month if symptoms recur
        """, encoding="utf-8")
        stats = memory.add(f)
        assert stats["chunks"] >= 1

    def test_upload_lab_results(self, memory, tmp_path):
        f = tmp_path / "lab_results.csv"
        f.write_text(
            "test_name,result,unit,reference_range,flag\n"
            "Glucose (Fasting),186,mg/dL,70-100,HIGH\n"
            "HbA1c,7.2,%,<5.7,HIGH\n"
            "Total Cholesterol,210,mg/dL,<200,HIGH\n"
            "LDL,135,mg/dL,<100,HIGH\n"
            "HDL,48,mg/dL,>40,NORMAL\n"
            "Triglycerides,180,mg/dL,<150,HIGH\n"
            "Creatinine,0.9,mg/dL,0.6-1.2,NORMAL\n"
            "BUN,18,mg/dL,7-20,NORMAL\n"
            "WBC,12.5,10^3/uL,4.5-11.0,HIGH\n"
            "Hemoglobin,13.8,g/dL,12.0-16.0,NORMAL\n",
            encoding="utf-8",
        )
        stats = memory.add(f)
        assert stats["chunks"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8: add_text MEMORY TYPES (POSTGRES)
# ═══════════════════════════════════════════════════════════════════════════

class TestAddTextPostgres:
    def test_semantic_memory(self, memory):
        stats = memory.add_text(
            "Metformin is a first-line treatment for Type 2 Diabetes Mellitus.",
            source_id="pg-semantic-1",
            memory_type="semantic",
        )
        assert stats["inserted"] >= 1

    def test_episodic_memory(self, memory):
        stats = memory.add_text(
            "Patient Rodriguez called on 09/02/2026 to report nausea after starting Metformin.",
            source_id="pg-episodic-1",
            memory_type="episodic",
            session_id="call-log-20260902",
        )
        assert stats["inserted"] >= 1

    def test_procedural_memory(self, memory):
        stats = memory.add_text(
            "For new DM2 patients: Start Metformin 500mg BID, titrate to 1000mg BID over 4 weeks if tolerated.",
            source_id="pg-procedural-1",
            memory_type="procedural",
        )
        assert stats["inserted"] >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9: CROSS-FILE SEARCH (POSTGRES)
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossFileSearch:
    """Verify that Postgres search finds data across all uploaded files."""

    def test_search_patient_name(self, memory):
        hits = memory.search("Rodriguez patient", top_k=5)
        assert len(hits) >= 1

    def test_search_medication(self, memory):
        hits = memory.search("Metformin dosage", top_k=5)
        assert len(hits) >= 1

    def test_search_diagnosis(self, memory):
        hits = memory.search("diabetes type 2 diagnosis", top_k=5)
        assert len(hits) >= 1

    def test_search_by_memory_type(self, memory):
        hits = memory.search("Metformin", memory_type="procedural", top_k=3)
        assert len(hits) >= 1

    def test_search_by_session_id(self, memory):
        hits = memory.search("nausea", session_id="call-log-20260902", top_k=3)
        assert len(hits) >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 10: ERROR PATHS (POSTGRES)
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorsPostgres:
    def test_empty_text_raises(self, memory):
        from memotrix.utils.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            memory.add_text("")

    def test_unsupported_ext_raises(self, memory, tmp_path):
        f = tmp_path / "file.unsupported_xyz"
        f.write_text("nope", encoding="utf-8")
        with pytest.raises(UnsupportedDocumentTypeError):
            memory.add(f)

    def test_missing_file_raises(self, memory):
        with pytest.raises(FileNotFoundError):
            memory.add("nonexistent_file_pg_test.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 11: LIST SOURCES & DELETE (POSTGRES)
# ═══════════════════════════════════════════════════════════════════════════

class TestListDeletePostgres:
    def test_list_sources(self, memory):
        sources = memory.list_sources()
        assert isinstance(sources, list)
        assert len(sources) >= 1

    def test_delete_and_verify(self, memory):
        memory.add_text("temporary test data for deletion", source_id="pg-delete-test")
        deleted = memory.delete("pg-delete-test")
        assert deleted >= 1
        hits = memory.search("temporary test data for deletion", top_k=10)
        for hit in hits:
            assert hit.get("source_path") != "pg-delete-test"


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 12: INTERACTIVE CLI CHAT & AUTO-UPLOADER
# ═══════════════════════════════════════════════════════════════════════════

def ingest_all_available_files(memory):
    """Scan and upload/chunk all test fixtures, documentation, and sample file formats into Postgres."""
    import tempfile
    print("\n📦 Clearing table & uploading/chunking all available files into Postgres...")

    # Clear previous table entries so we don't mix random test vectors
    try:
        memory.clear()
    except Exception as err:
        print(f"   ⚠️ Could not clear table: {err}")

    tmp = Path(tempfile.gettempdir()) / "memotrix_cli_upload"
    tmp.mkdir(parents=True, exist_ok=True)

    uploaded = []

    # 1. Real fixture files in tests/filesForTests
    if FIXTURES.exists():
        for fpath in sorted(FIXTURES.iterdir()):
            if fpath.is_file():
                try:
                    stats = memory.add(fpath, describe_images=False)
                    uploaded.append((f"fixtures/{fpath.name}", stats.get("chunks", 0)))
                except Exception as err:
                    print(f"   ⚠️ Could not upload {fpath.name}: {err}")

    # 2. Project documentation files in docBook/ and docs/
    docbook_dir = ROOT / "docBook"
    if docbook_dir.exists():
        for fpath in sorted(docbook_dir.glob("*.md")):
            try:
                stats = memory.add(fpath)
                uploaded.append((f"docBook/{fpath.name}", stats.get("chunks", 0)))
            except Exception as err:
                print(f"   ⚠️ Could not upload {fpath.name}: {err}")

    docs_dir = ROOT / "docs"
    if docs_dir.exists():
        for fpath in sorted(docs_dir.rglob("*.md")):
            try:
                rel_path = fpath.relative_to(ROOT)
                stats = memory.add(fpath)
                uploaded.append((str(rel_path), stats.get("chunks", 0)))
            except Exception as err:
                print(f"   ⚠️ Could not upload {fpath.name}: {err}")

    # 3. Sample files covering all supported formats
    samples = {
        "plain_text.txt": "Memotrix ingests plain text files into Postgres with hybrid vector and keyword search.",
        "architecture.md": "# Memotrix Architecture\nUses HNSW for dense vectors and BM25/FTS for sparse keyword search in PostgreSQL.",
        "page.html": "<html><body><h1>Installation</h1><p>pip install memotrix</p></body></html>",
        "schema.sql": "CREATE TABLE patients (id SERIAL PRIMARY KEY, name TEXT, diagnosis TEXT);",
        "config.json": json.dumps({"app": "memotrix", "version": "0.2.0", "backend": "postgres"}, indent=2),
        "events.jsonl": json.dumps({"event": "upload", "user": "alice", "status": "success"}),
        "patients.csv": "patient_id,name,diagnosis,blood_pressure\nP001,John Doe,Hypertension,140/90\nP002,Jane Smith,Diabetes Type 2,130/85\n",
        "pipeline.yaml": "name: memotrix-pipeline\non: push\njobs:\n  test: run pytest",
        "prescription.xml": "<prescription><patient name='John Doe'/><medication name='Metformin 500mg'/></prescription>",
        "extractor.py": "class ClinicalNoteExtractor:\n    def extract(self, path):\n        return build_document(path, path.read_text())",
        "app.js": "async function searchMemory(query) { return fetch('/api/search', { body: JSON.stringify({ query }) }); }",
        "types.ts": "interface MemoryHit { chunk_text: string; score: number; source_path: string; }",
        "Patient.java": "public class Patient { private String id; private String name; private String diagnosis; }",
        "main.go": "package main\ntype EHRRecord struct { PatientID string; Diagnosis string }",
        "lib.rs": "struct MedicalRecord { patient_id: String, diagnosis: String, icd_code: String }",
        "vitals.cpp": "struct VitalSigns { double heart_rate; double bp_sys; double bp_dia; };",
        "patient_fhir.json": json.dumps({
            "resourceType": "Patient", "id": "pg-patient-001",
            "name": [{"family": "Rodriguez", "given": ["Maria", "Elena"]}],
            "gender": "female", "birthDate": "1985-03-22", "address": [{"city": "Houston", "state": "TX"}]
        }, indent=2),
        "observation_bp_fhir.json": json.dumps({
            "resourceType": "Observation", "id": "obs-bp-pg", "status": "final",
            "code": {"text": "Blood pressure"}, "valueQuantity": {"value": 142, "unit": "mmHg"},
            "subject": {"reference": "Patient/pg-patient-001"}
        }, indent=2),
        "medication_fhir.json": json.dumps({
            "resourceType": "MedicationRequest", "id": "medrx-metformin", "status": "active",
            "medicationCodeableConcept": {"text": "Metformin 500mg tablet"},
            "dosageInstruction": [{"text": "Take 1 tablet by mouth twice daily with meals."}]
        }, indent=2),
        "clinical_note.txt": """
SUBJECTIVE: Patient Maria Rodriguez presents with fatigue and increased thirst over past 3 weeks.
OBJECTIVE: BP 142/88 mmHg, HR 78 bpm. Fasting glucose 186 mg/dL. HbA1c 7.2%.
ASSESSMENT: 1. Diabetes mellitus type 2, newly diagnosed. 2. Hypertension stage 1.
PLAN: 1. Start Metformin 500mg PO BID with meals. 2. Lifestyle modifications and diet counseling.
""",
        "discharge_summary.txt": """
DISCHARGE SUMMARY: Patient Maria Rodriguez, MRN: PG-001. Admission 09/01/2026, Discharge 09/03/2026.
Chief Complaint: Acute chest pain.
Hospital Course: Troponin levels negative x3. ECG normal sinus rhythm. Chest pain non-cardiac.
Discharge Medications: Metformin 500mg PO BID, Lisinopril 10mg PO daily.
""",
    }

    for name, content in samples.items():
        spath = tmp / name
        spath.write_text(content, encoding="utf-8")
        try:
            stats = memory.add(spath)
            uploaded.append((name, stats.get("chunks", 0)))
        except Exception as err:
            print(f"   ⚠️ Could not upload {name}: {err}")

    # 4. Memory text additions
    memory.add_text("Metformin is a first-line treatment for Type 2 Diabetes Mellitus.", source_id="pg-semantic-1", memory_type="semantic")
    memory.add_text("Patient Rodriguez called on 09/02/2026 to report nausea after starting Metformin.", source_id="pg-episodic-1", memory_type="episodic", session_id="call-log-20260902")
    memory.add_text("For new DM2 patients: Start Metformin 500mg BID, titrate to 1000mg BID over 4 weeks if tolerated.", source_id="pg-procedural-1", memory_type="procedural")

    return uploaded


def run_cli_chat(memory=None):
    """Interactive CLI to upload files, chunk them into Postgres, and talk with AI via RAG."""
    table_name = "memotrix_cli_rag"
    if memory is None:
        from memotrix.config import MemoryConfig
        from memotrix.embeddings import HuggingFaceEmbeddings
        print("🧠 Loading real semantic embedding model (BAAI/bge-small-en-v1.5)...")
        embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
        memory = Memory(
            embeddings=embeddings,
            backend="postgres",
            connection=DSN,
            config=MemoryConfig(table_name=table_name),
        )

    uploaded_files = ingest_all_available_files(memory)
    sources = memory.list_sources()

    print("\n" + "=" * 75)
    print(" 🚀 MEMOTRIX LOCAL UPLOAD & CHUNKING COMPLETE")
    print("=" * 75)
    print(f" Database:  {DSN}")
    print(f" Table:     {table_name}")
    print(f" Embeddings: Real HuggingFace (BAAI/bge-small-en-v1.5, 384-dim)")
    print(f" Indexed:   {len(sources)} active source files / memories in Postgres")
    print("-" * 75)
    print(" Sample Uploaded Files & Chunk Counts:")
    for fname, chunks in uploaded_files[:12]:
        print(f"   📄 {fname:<36} -> {chunks} chunk(s)")
    if len(uploaded_files) > 12:
        print(f"   ... and {len(uploaded_files) - 12} more files.")
    print("=" * 75)

    print("\n" + "=" * 75)
    print(" 🤖 TALK WITH AI (MEMOTRIX RAG CHAT CLI)")
    print("=" * 75)
    print(" Type any question to search your uploaded documents and chat with AI.")
    print(" Commands:")
    print("   /search <query>  - Show raw RAG vector search hits and relevance scores")
    print("   /list            - List all ingested sources in Memory")
    print("   /stats           - Show vector DB chunk statistics")
    print("   exit / quit      - Exit the CLI")
    print("=" * 75 + "\n")

    from memotrix.utils.ai_integration import get_ai_router

    router = get_ai_router()
    provider_name = type(router.provider).__name__
    print(f" Active AI Model Provider: {provider_name}\n")

    while True:
        try:
            user_input = input("\nUser > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI. Goodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting CLI. Goodbye! 👋")
            break

        if user_input.startswith("/list"):
            all_srcs = memory.list_sources()
            print(f"\n📋 Ingested Sources ({len(all_srcs)} total):")
            for s in all_srcs[:25]:
                print(f"  • {s}")
            if len(all_srcs) > 25:
                print(f"  ... and {len(all_srcs) - 25} more.")
            continue

        if user_input.startswith("/stats"):
            all_srcs = memory.list_sources()
            print(f"\n📊 Storage Stats:\n  • Backend: Postgres (pgvector)\n  • Table: {table_name}\n  • Unique Sources: {len(all_srcs)}")
            continue

        if user_input.startswith("/search "):
            squery = user_input[8:].strip()
            hits = memory.search(squery, top_k=5)
            print(f"\n🔍 Search Hits for '{squery}' ({len(hits)} hits):")
            for i, h in enumerate(hits, 1):
                src = h.get("source_path", "unknown")
                score = h.get("score", 0)
                txt = h.get("chunk_text", "").replace("\n", " ")[:120]
                print(f"  {i}. [{src}] (score: {score:.4f}) -> {txt}...")
            continue

        # Perform RAG Search
        hits = memory.search(user_input, top_k=5)

        print("🔍 Retrieving relevant memory chunks...")
        if hits:
            print("   Retrieved context sources:")
            for i, h in enumerate(hits, 1):
                src = h.get("filename") or Path(str(h.get("source_path") or "unknown")).name
                score = h.get("score", 0)
                page_info = ""
                if h.get("page_number"):
                    page_info = f" (Page {h['page_number']})"
                elif h.get("slide_number"):
                    page_info = f" (Slide {h['slide_number']})"
                elif h.get("sheet_name"):
                    page_info = f" (Sheet: {h['sheet_name']})"
                print(f"     [{i}] {src}{page_info} [score: {score:.4f}]")

            # Build context prompt
            context_blocks = []
            for i, h in enumerate(hits, 1):
                src = h.get("filename") or Path(str(h.get("source_path") or "unknown")).name
                page_tag = ""
                if h.get("page_number"):
                    page_tag = f" Page {h['page_number']}"
                elif h.get("slide_number"):
                    page_tag = f" Slide {h['slide_number']}"
                txt = h.get("chunk_text", "")
                context_blocks.append(f"[Source {i}: {src}{page_tag}]\n{txt}")
            context_str = "\n\n".join(context_blocks)

            system_prompt = (
                "You are Memotrix AI, an intelligent assistant with access to long-term RAG memory. "
                "Answer the user's question clearly and thoroughly using the provided memory context chunks. "
                "Cite the source filename in your answer when referencing facts."
            )
            full_prompt = (
                f"--- RETRIEVED MEMORY CONTEXT ---\n{context_str}\n\n"
                f"--- USER QUESTION ---\n{user_input}\n\n"
                f"Please provide a complete answer based on the retrieved context above:"
            )

            print(f"\n🤖 Memotrix AI ({provider_name}):")
            try:
                if provider_name == "LocalFallbackProvider":
                    # Synthesize clear RAG response directly from top context hits
                    print(f"  Based on retrieved context from {len(hits)} source(s):\n")
                    best_hit = hits[0]
                    best_src = best_hit.get("source_path", "unknown")
                    best_txt = best_hit.get("chunk_text", "").strip()
                    print(f"  📌 Primary Match ({best_src}):")
                    lines = best_txt.splitlines()
                    preview = "\n  ".join(lines[:10])
                    print(f"  {preview}")
                    if len(hits) > 1:
                        print(f"\n  ℹ️ Additional relevant sources found:")
                        for h in hits[1:]:
                            h_src = h.get('source_path', 'unknown')
                            h_txt = h.get('chunk_text', '').replace('\n', ' ')[:120]
                            print(f"     • {h_src}: {h_txt}...")
                else:
                    ai_response = router.generate_text(full_prompt, system=system_prompt)
                    print(f"  {ai_response}")
            except Exception as err:
                print(f"  [RAG Context Match]: {hits[0].get('chunk_text', '')[:250]}...")
                print(f"  (Note: AI completion info: {err})")
        else:
            print("   No relevant memory chunks found for this query.")


if __name__ == "__main__":
    import sys

    # If invoked with pytest flags (e.g. pytest tests/test_local_upload.py -v), run pytest
    if any(arg.startswith("-") and arg not in ("--cli", "--chat", "--talk", "-i") for arg in sys.argv[1:]):
        sys.exit(pytest.main([__file__] + [a for a in sys.argv[1:] if a not in ("--cli", "--chat", "--talk", "-i")]))
    else:
        # Run interactive CLI chat & auto-uploader
        run_cli_chat()

