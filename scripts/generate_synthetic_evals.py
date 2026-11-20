import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"
OUTPUT_FILE = DATA_DIR / "ground_truth_100.json"

# Base QA templates spanning the 3 corporate policy documents + negative out-of-scope controls
QA_TEMPLATES = [
    # --- Employee Handbook (Working Hours, Remote Work, Leave, Budget) ---
    {
        "q": "What are the core working hours at TechNova?",
        "doc": "technova_employee_handbook.pdf", "sec": "Working Hours",
        "keys": ["10:00 to 16:00", "core hours"],
        "gt": "Core hours are 10:00 to 16:00 in the employee's local time zone."
    },
    {
        "q": "How many hours per week are employees expected to work on average?",
        "doc": "technova_employee_handbook.pdf", "sec": "Working Hours",
        "keys": ["40 hours"],
        "gt": "Employees are expected to work an average of 40 hours per week."
    },
    {
        "q": "What is the policy for meetings held outside core hours?",
        "doc": "technova_employee_handbook.pdf", "sec": "Working Hours",
        "keys": ["recorded"],
        "gt": "Meetings outside core hours should be recorded so remote colleagues can catch up."
    },
    {
        "q": "How many days per week are employees allowed to work remotely?",
        "doc": "technova_employee_handbook.pdf", "sec": "Remote Work",
        "keys": ["four days", "4 days"],
        "gt": "Employees may work remotely up to four days per week."
    },
    {
        "q": "What must an employee do if working from another country for over 30 days?",
        "doc": "technova_employee_handbook.pdf", "sec": "Remote Work",
        "keys": ["30 days", "People team", "tax obligations"],
        "gt": "Notify the People team first due to potential tax obligations for both parties."
    },
    {
        "q": "How many days of paid annual leave do full-time employees accrue per year?",
        "doc": "technova_employee_handbook.pdf", "sec": "Annual Leave",
        "keys": ["24 days", "two days per month"],
        "gt": "Full-time employees receive 24 days of paid annual leave accrued at 2 days per month."
    },
    {
        "q": "How many unused annual leave days can be carried over into the next year?",
        "doc": "technova_employee_handbook.pdf", "sec": "Annual Leave",
        "keys": ["five", "5", "31 March"],
        "gt": "Up to five unused leave days may be carried over and must be used before 31 March."
    },
    {
        "q": "Within how many days are annual leave requests normally approved?",
        "doc": "technova_employee_handbook.pdf", "sec": "Annual Leave",
        "keys": ["three working days"],
        "gt": "Leave requests are submitted in the People portal and approved within three working days."
    },
    {
        "q": "How many paid sick leave days are provided each year?",
        "doc": "technova_employee_handbook.pdf", "sec": "Sick Leave",
        "keys": ["10 days"],
        "gt": "Employees receive up to 10 days of paid sick leave per year."
    },
    {
        "q": "When is a medical certificate required for sick leave?",
        "doc": "technova_employee_handbook.pdf", "sec": "Sick Leave",
        "keys": ["three consecutive", "3 consecutive"],
        "gt": "A medical certificate is required for absences longer than three consecutive working days."
    },
    {
        "q": "Do mental health days count as sick leave?",
        "doc": "technova_employee_handbook.pdf", "sec": "Sick Leave",
        "keys": ["mental health", "count as sick leave"],
        "gt": "Yes, mental health days count as sick leave and reasons do not need to be explained."
    },
    {
        "q": "What is the paid parental leave duration for primary and secondary caregivers?",
        "doc": "technova_employee_handbook.pdf", "sec": "Parental Leave",
        "keys": ["16 weeks", "6 weeks"],
        "gt": "16 weeks fully paid for primary caregivers and 6 weeks for secondary caregivers."
    },
    {
        "q": "Can parental leave be taken in separate blocks?",
        "doc": "technova_employee_handbook.pdf", "sec": "Parental Leave",
        "keys": ["three separate blocks", "18 months"],
        "gt": "Parental leave can be taken in up to three separate blocks during the first 18 months."
    },
    {
        "q": "What part-time arrangement is offered to employees returning from parental leave?",
        "doc": "technova_employee_handbook.pdf", "sec": "Parental Leave",
        "keys": ["four-day week", "80 percent", "three months"],
        "gt": "Returning employees may work a reduced four-day week at 80% pay for the first 3 months."
    },
    {
        "q": "What is the annual learning budget provided to each employee?",
        "doc": "technova_employee_handbook.pdf", "sec": "Learning Budget",
        "keys": ["1500 USD"],
        "gt": "Each employee has an annual learning budget of 1500 USD."
    },
    {
        "q": "Does the annual learning budget carry over to the next year?",
        "doc": "technova_employee_handbook.pdf", "sec": "Learning Budget",
        "keys": ["does not carry over"],
        "gt": "No, the learning budget does not carry over to the following year."
    },
    {
        "q": "What purchase threshold requires manager pre-approval for the learning budget?",
        "doc": "technova_employee_handbook.pdf", "sec": "Learning Budget",
        "keys": ["500 USD", "manager approval"],
        "gt": "Purchases above 500 USD need manager approval in advance."
    },
    {
        "q": "By what day of the month must expense reimbursement claims be submitted?",
        "doc": "technova_employee_handbook.pdf", "sec": "Expense Reimbursement",
        "keys": ["25th"],
        "gt": "Claims must be submitted before the 25th of the month to be paid with that month's payroll."
    },
    {
        "q": "What flight duration qualifies for business class travel?",
        "doc": "technova_employee_handbook.pdf", "sec": "Expense Reimbursement",
        "keys": ["six hours", "director approval"],
        "gt": "Flights over six hours may be booked in business class with director approval."
    },
    {
        "q": "What computer options do engineers receive upon joining TechNova?",
        "doc": "technova_employee_handbook.pdf", "sec": "Equipment Policy",
        "keys": ["MacBook Pro", "Dell XPS", "Ubuntu"],
        "gt": "Engineers may choose between a MacBook Pro and a Dell XPS running Ubuntu."
    },
    {
        "q": "What is the hardware refresh cycle for company laptops?",
        "doc": "technova_employee_handbook.pdf", "sec": "Equipment Policy",
        "keys": ["three years"],
        "gt": "Laptops are refreshed every three years."
    },
    {
        "q": "What is the home office allowance amount for new hires?",
        "doc": "technova_employee_handbook.pdf", "sec": "Equipment Policy",
        "keys": ["600 USD"],
        "gt": "Employees receive a one-time home office allowance of 600 USD for a desk and chair."
    },
    {
        "q": "When do performance review cycles occur at TechNova?",
        "doc": "technova_employee_handbook.pdf", "sec": "Performance Reviews",
        "keys": ["March and September"],
        "gt": "Performance reviews happen twice a year, in March and September."
    },
    {
        "q": "When do compensation changes and promotions take effect?",
        "doc": "technova_employee_handbook.pdf", "sec": "Performance Reviews",
        "keys": ["1 May", "March"],
        "gt": "Decisions are made in the March cycle and take effect on 1 May."
    },
    {
        "q": "Within how many days are Code of Conduct reports investigated?",
        "doc": "technova_employee_handbook.pdf", "sec": "Code of Conduct",
        "keys": ["10 working days"],
        "gt": "All reports are investigated within 10 working days."
    },

    # --- Product Guide (NovaCloud architecture, pricing, limits, APIs) ---
    {
        "q": "What is NovaCloud and what is its primary use case?",
        "doc": "technova_product_guide.pdf", "sec": "NovaCloud Overview",
        "keys": ["data pipeline", "databases", "storage"],
        "gt": "NovaCloud is a platform to build, schedule, and monitor pipelines moving data without managing servers."
    },
    {
        "q": "What are the three main components of NovaCloud?",
        "doc": "technova_product_guide.pdf", "sec": "NovaCloud Overview",
        "keys": ["Pipeline Builder", "Scheduler", "Observability Hub"],
        "gt": "The platform has three parts: Pipeline Builder, Scheduler, and Observability Hub."
    },
    {
        "q": "What is the current release of NovaCloud and when did it ship?",
        "doc": "technova_product_guide.pdf", "sec": "NovaCloud Overview",
        "keys": ["4.2", "February 2024"],
        "gt": "The current release is NovaCloud 4.2, shipped in February 2024."
    },
    {
        "q": "In what languages can transforms be written in the Pipeline Builder?",
        "doc": "technova_product_guide.pdf", "sec": "Pipeline Builder",
        "keys": ["SQL", "Python"],
        "gt": "Transforms in the Pipeline Builder are written in SQL or Python."
    },
    {
        "q": "What execution time and memory limits apply to Python transforms?",
        "doc": "technova_product_guide.pdf", "sec": "Pipeline Builder",
        "keys": ["10 minute", "4 GB"],
        "gt": "Python transforms run in an isolated sandbox with a 10 minute execution limit and 4 GB of memory."
    },
    {
        "q": "How can pipelines be exported from the visual builder?",
        "doc": "technova_product_guide.pdf", "sec": "Pipeline Builder",
        "keys": ["YAML", "Git"],
        "gt": "Every pipeline can be exported as a YAML file and stored in Git."
    },
    {
        "q": "What retry policy does the Scheduler use for failed pipeline runs?",
        "doc": "technova_product_guide.pdf", "sec": "Scheduler",
        "keys": ["three times", "exponential backoff"],
        "gt": "Failed runs are retried three times with exponential backoff before being marked failed."
    },
    {
        "q": "How long are pipeline metrics retained in the Observability Hub?",
        "doc": "technova_product_guide.pdf", "sec": "Observability Hub",
        "keys": ["13 months"],
        "gt": "Metrics are retained for 13 months on all paid plans."
    },
    {
        "q": "What data quality checks are available in the Observability Hub?",
        "doc": "technova_product_guide.pdf", "sec": "Observability Hub",
        "keys": ["null checks", "uniqueness", "range", "freshness"],
        "gt": "Checks include null checks, uniqueness checks, range checks, and freshness checks."
    },
    {
        "q": "How many pre-built connectors ship with NovaCloud 4.2?",
        "doc": "technova_product_guide.pdf", "sec": "Connectors",
        "keys": ["68 connectors"],
        "gt": "NovaCloud 4.2 ships with 68 pre-built connectors."
    },
    {
        "q": "Can customers build private custom connectors for NovaCloud?",
        "doc": "technova_product_guide.pdf", "sec": "Connectors",
        "keys": ["Connector SDK", "private connector"],
        "gt": "Yes, customers can build private connectors using the Connector SDK visible only in their workspace."
    },
    {
        "q": "What is the monthly price and pipeline allowance of the Starter tier?",
        "doc": "technova_product_guide.pdf", "sec": "Pricing Tiers",
        "keys": ["99 USD", "3 pipelines", "10 GB"],
        "gt": "Starter costs 99 USD per month and includes 3 pipelines, 1 workspace, and 10 GB transfer."
    },
    {
        "q": "What is included in the Growth subscription tier?",
        "doc": "technova_product_guide.pdf", "sec": "Pricing Tiers",
        "keys": ["499 USD", "25 pipelines", "250 GB"],
        "gt": "Growth costs 499 USD per month and includes 25 pipelines, 3 workspaces, and 250 GB data transfer."
    },
    {
        "q": "What is the overage rate for data transfer above the tier allowance?",
        "doc": "technova_product_guide.pdf", "sec": "Pricing Tiers",
        "keys": ["0.09 USD per GB"],
        "gt": "Data transfer above the included allowance is billed at 0.09 USD per GB."
    },
    {
        "q": "What is the maximum number of steps allowed in a single pipeline?",
        "doc": "technova_product_guide.pdf", "sec": "Platform Limits",
        "keys": ["150 steps"],
        "gt": "A single pipeline may contain up to 150 steps."
    },
    {
        "q": "What is the maximum file size that can be processed in a single step?",
        "doc": "technova_product_guide.pdf", "sec": "Platform Limits",
        "keys": ["5 GB"],
        "gt": "The maximum size of one file processed in a single step is 5 GB."
    },
    {
        "q": "What is the NovaCloud REST API rate limit per workspace?",
        "doc": "technova_product_guide.pdf", "sec": "Platform Limits",
        "keys": ["600 requests per minute"],
        "gt": "The REST API is rate limited to 600 requests per minute per workspace."
    },
    {
        "q": "What notification channels are supported for pipeline alerts?",
        "doc": "technova_product_guide.pdf", "sec": "Integrations and Alerting",
        "keys": ["email", "Slack", "Teams", "PagerDuty", "webhook"],
        "gt": "Alerts support email, Slack, Microsoft Teams, PagerDuty, and HTTPS webhooks."
    },
    {
        "q": "Does NovaCloud expose Prometheus metrics?",
        "doc": "technova_product_guide.pdf", "sec": "Integrations and Alerting",
        "keys": ["Prometheus"],
        "gt": "Yes, NovaCloud exposes a Prometheus-compatible metrics endpoint for customer monitoring."
    },
    {
        "q": "How long is the free trial period and what tier does it include?",
        "doc": "technova_product_guide.pdf", "sec": "Getting Started",
        "keys": ["14 days", "Growth tier"],
        "gt": "Free trials last 14 days and include all Growth tier features with no credit card required."
    },
    {
        "q": "When is NovaCloud REST API version v1 scheduled for sunset?",
        "doc": "technova_product_guide.pdf", "sec": "Deprecation and Versioning",
        "keys": ["31 December 2024", "v1"],
        "gt": "Version v1 is deprecated and will be switched off on 31 December 2024."
    },
    {
        "q": "How much advance notice does TechNova provide before removing an API version?",
        "doc": "technova_product_guide.pdf", "sec": "Deprecation and Versioning",
        "keys": ["12 months notice"],
        "gt": "TechNova gives at least 12 months notice before removing an API version."
    },

    # --- Security Policy (Passwords, MFA, Access, Encryption, Incidents) ---
    {
        "q": "Who approves the Information Security Policy and how often is it reviewed?",
        "doc": "technova_security_policy.docx", "sec": "Purpose and Scope",
        "keys": ["CTO", "12 months"],
        "gt": "It is reviewed every 12 months by the Security team and approved by the CTO."
    },
    {
        "q": "What is TechNova's minimum password length?",
        "doc": "technova_security_policy.docx", "sec": "Password Policy",
        "keys": ["14 characters"],
        "gt": "Passwords must be at least 14 characters long."
    },
    {
        "q": "What password manager is mandatory for employees?",
        "doc": "technova_security_policy.docx", "sec": "Password Policy",
        "keys": ["1Password"],
        "gt": "All employees must store work passwords in the company password manager, 1Password."
    },
    {
        "q": "Are passwords rotated on a routine schedule?",
        "doc": "technova_security_policy.docx", "sec": "Password Policy",
        "keys": ["not rotated", "NIST"],
        "gt": "No, following NIST guidance, password changes are only forced upon evidence of compromise."
    },
    {
        "q": "Can employees share their work passwords with colleagues?",
        "doc": "technova_security_policy.docx", "sec": "Password Policy",
        "keys": ["never permitted", "not permitted"],
        "gt": "Sharing passwords with anyone, including colleagues, is strictly prohibited."
    },
    {
        "q": "Are SMS codes accepted as a multi-factor authentication method?",
        "doc": "technova_security_policy.docx", "sec": "Multi-Factor Authentication",
        "keys": ["SMS", "not accepted", "SIM swapping"],
        "gt": "SMS codes are not accepted due to the risk of SIM swapping attacks."
    },
    {
        "q": "What MFA methods are required for production systems?",
        "doc": "technova_security_policy.docx", "sec": "Multi-Factor Authentication",
        "keys": ["hardware security keys", "authenticator app"],
        "gt": "Hardware security keys or an authenticator app must be used."
    },
    {
        "q": "What default access level do engineers receive to production systems?",
        "doc": "technova_security_policy.docx", "sec": "Access Control",
        "keys": ["read-only", "least privilege"],
        "gt": "Engineers receive read-only access by default under the principle of least privilege."
    },
    {
        "q": "For how long is just-in-time write access granted to production?",
        "doc": "technova_security_policy.docx", "sec": "Access Control",
        "keys": ["eight hours", "8 hours"],
        "gt": "Write access to production is granted for a maximum of eight hours via the JIT tool."
    },
    {
        "q": "Within how many hours must departed employee accounts be disabled?",
        "doc": "technova_security_policy.docx", "sec": "Access Control",
        "keys": ["two hours", "2 hours"],
        "gt": "Accounts of employees who leave are disabled within two hours of their final working day."
    },
    {
        "q": "What encryption standard protects customer data in transit?",
        "doc": "technova_security_policy.docx", "sec": "Data Encryption",
        "keys": ["TLS 1.2"],
        "gt": "Customer data is encrypted in transit using TLS 1.2 or higher."
    },
    {
        "q": "What encryption algorithm is used for customer data at rest?",
        "doc": "technova_security_policy.docx", "sec": "Data Encryption",
        "keys": ["AES-256"],
        "gt": "Customer data is encrypted at rest with AES-256."
    },
    {
        "q": "How frequently are customer encryption keys rotated?",
        "doc": "technova_security_policy.docx", "sec": "Data Encryption",
        "keys": ["rotated annually"],
        "gt": "Encryption keys are managed in KMS and rotated annually."
    },
    {
        "q": "Is customer data ever permitted to be copied to personal laptops?",
        "doc": "technova_security_policy.docx", "sec": "Data Encryption",
        "keys": ["never copied"],
        "gt": "Customer data is never copied to a laptop or personal device."
    },
    {
        "q": "After how many minutes of inactivity must company laptop screens lock?",
        "doc": "technova_security_policy.docx", "sec": "Device Security",
        "keys": ["five minutes", "5 minutes"],
        "gt": "Automatic screen lock activates after five minutes of inactivity."
    },
    {
        "q": "Within how many days must operating system security patches be installed?",
        "doc": "technova_security_policy.docx", "sec": "Device Security",
        "keys": ["14 days"],
        "gt": "Employees must install OS security updates within 14 days of release."
    },
    {
        "q": "Within what timeframe must a suspected security incident be reported?",
        "doc": "technova_security_policy.docx", "sec": "Incident Response",
        "keys": ["one hour", "1 hour"],
        "gt": "Anyone suspecting a security incident must report it within one hour."
    },
    {
        "q": "Under GDPR, what is the notification window for confirmed personal data breaches?",
        "doc": "technova_security_policy.docx", "sec": "Incident Response",
        "keys": ["72 hours", "GDPR"],
        "gt": "Affected customers and supervisory authorities must be notified within 72 hours."
    },
    {
        "q": "What security compliance certifications does TechNova maintain?",
        "doc": "technova_security_policy.docx", "sec": "Compliance and Certification",
        "keys": ["SOC 2 Type II", "ISO 27001"],
        "gt": "TechNova maintains an annual SOC 2 Type II report and is ISO 27001 certified."
    },
    {
        "q": "How frequently are third-party penetration tests carried out?",
        "doc": "technova_security_policy.docx", "sec": "Compliance and Certification",
        "keys": ["twice a year"],
        "gt": "Penetration tests are conducted by an external firm twice a year."
    },
    {
        "q": "How often is security awareness training required for all employees?",
        "doc": "technova_security_policy.docx", "sec": "Security Training",
        "keys": ["12 months", "onboarding"],
        "gt": "Training is completed during onboarding and refreshed every 12 months."
    },

    # --- Negative Controls / Out of Scope (Tests Hallucination & Abstention) ---
    {
        "q": "Does TechNova offer catered lunch or free meals in the Dublin office?",
        "doc": None, "sec": None,
        "keys": ["not", "does not contain", "no information"],
        "gt": "The documentation does not contain information regarding catered lunch."
    },
    {
        "q": "What is the 401(k) company matching percentage for US employees?",
        "doc": None, "sec": None,
        "keys": ["not", "does not mention", "no information"],
        "gt": "The documents do not mention 401(k) matching percentages."
    },
    {
        "q": "Are employees allowed to bring dogs or pets into the Austin headquarters?",
        "doc": None, "sec": None,
        "keys": ["not", "no information"],
        "gt": "There is no pet policy documented in the provided files."
    },
    {
        "q": "Does TechNova provide company cars or vehicle stipends for sales staff?",
        "doc": None, "sec": None,
        "keys": ["not", "does not mention"],
        "gt": "No company vehicle policy is mentioned in the company documents."
    },
    {
        "q": "What is the stock vesting schedule for newly granted employee ISO options?",
        "doc": None, "sec": None,
        "keys": ["not", "no information"],
        "gt": "Equity and stock option vesting schedules are not documented in the indexed files."
    },
    {
        "q": "Is there an on-site gym or wellness center at the Singapore office?",
        "doc": None, "sec": None,
        "keys": ["not", "does not contain"],
        "gt": "The documentation does not cover on-site gym facilities."
    },
    {
        "q": "What is the relocation reimbursement package for international transfers?",
        "doc": None, "sec": None,
        "keys": ["not", "does not mention"],
        "gt": "Relocation packages are not detailed in the provided documents."
    },
    {
        "q": "Does TechNova provide childcare subsidies or on-site day care?",
        "doc": None, "sec": None,
        "keys": ["not", "no information"],
        "gt": "Childcare subsidies are not covered in the employee handbook."
    }
]

def generate_100_eval_cases():
    cases = []
    case_counter = 1

    # First add all original base questions
    for item in QA_TEMPLATES:
        cases.append({
            "id": f"eval_{case_counter:03d}",
            "question": item["q"],
            "expected_document": item["doc"],
            "expected_section": item["sec"],
            "expected_key_phrases": item["keys"],
            "ground_truth_answer": item["gt"]
        })
        case_counter += 1

    # Add paraphrased / conversational variants to simulate realistic multi-query real users
    VARIATIONS = [
        ("Working hours summary", "Can you explain the daily schedule, core hours, and weekly expected hours?", "technova_employee_handbook.pdf", "Working Hours", ["10:00 to 16:00", "40 hours"]),
        ("Remote work abroad", "I want to work remotely from Spain for 2 months, is that allowed?", "technova_employee_handbook.pdf", "Remote Work", ["30 days", "People team", "tax"]),
        ("Vacation carryover", "Do my leftover vacation days expire at the end of December?", "technova_employee_handbook.pdf", "Annual Leave", ["31 March", "5", "five"]),
        ("Sick leave doctor note", "At what point do I have to provide a doctor's certificate for illness?", "technova_employee_handbook.pdf", "Sick Leave", ["three consecutive", "3"]),
        ("Parental leave blocks", "Can a father split his paternity leave into multiple periods?", "technova_employee_handbook.pdf", "Parental Leave", ["three separate blocks", "6 weeks"]),
        ("Conference ticket expense", "Can I use my education stipend for a $700 conference ticket?", "technova_employee_handbook.pdf", "Learning Budget", ["1500 USD", "manager approval", "500 USD"]),
        ("Travel flight booking", "Am I allowed to fly business class for an 8-hour transatlantic flight?", "technova_employee_handbook.pdf", "Expense Reimbursement", ["six hours", "director approval"]),
        ("Laptop OS options", "Can software engineers request a Linux OS laptop?", "technova_employee_handbook.pdf", "Equipment Policy", ["Ubuntu", "Dell XPS"]),
        ("Home desk reimbursement", "How much money do remote workers get for setting up a home desk?", "technova_employee_handbook.pdf", "Equipment Policy", ["600 USD"]),
        ("Promotion cycle timing", "When during the year do compensation reviews and promotions occur?", "technova_employee_handbook.pdf", "Performance Reviews", ["March", "1 May"]),
        ("Retaliation protection", "What happens if someone retaliates after an ethics hotline report?", "technova_employee_handbook.pdf", "Code of Conduct", ["serious breach", "retaliation"]),
        ("NovaCloud Python runtime limits", "What happens if my Python transform runs for 15 minutes?", "technova_product_guide.pdf", "Pipeline Builder", ["10 minute"]),
        ("Pipeline retry count", "If a pipeline fails due to a network glitch, how many times will it retry?", "technova_product_guide.pdf", "Scheduler", ["three times", "backoff"]),
        ("Metric history retention", "Can I see pipeline throughput metrics from 10 months ago?", "technova_product_guide.pdf", "Observability Hub", ["13 months"]),
        ("Snowflake BigQuery support", "Does NovaCloud support connectors to Snowflake and Google BigQuery?", "technova_product_guide.pdf", "Connectors", ["68 connectors", "Snowflake", "BigQuery"]),
        ("NovaCloud pricing comparison", "Compare the price and pipeline limits between Starter and Growth plans.", "technova_product_guide.pdf", "Pricing Tiers", ["99 USD", "499 USD", "3 pipelines", "25 pipelines"]),
        ("Data overage charge", "What fee is charged if a company exceeds its monthly gigabyte limit?", "technova_product_guide.pdf", "Pricing Tiers", ["0.09 USD per GB"]),
        ("Single file size maximum", "What is the heaviest single file that can pass through a pipeline step?", "technova_product_guide.pdf", "Platform Limits", ["5 GB"]),
        ("Webhook alert destinations", "Can NovaCloud send incident alerts to Microsoft Teams channels?", "technova_product_guide.pdf", "Integrations and Alerting", ["Teams", "webhook"]),
        ("API deprecation timeline", "How long will NovaCloud v1 REST API remain active?", "technova_product_guide.pdf", "Deprecation and Versioning", ["31 December 2024"]),
        ("Password length rule", "What is the minimum number of characters required for TechNova logins?", "technova_security_policy.docx", "Password Policy", ["14 characters"]),
        ("Mandatory password vault", "Which password manager is required by company security guidelines?", "technova_security_policy.docx", "Password Policy", ["1Password"]),
        ("MFA authenticator app", "Can I use SMS OTPs for my VPN login?", "technova_security_policy.docx", "Multi-Factor Authentication", ["SMS", "not accepted"]),
        ("Temporary write access", "For how long can an engineer get temporary write privileges in production?", "technova_security_policy.docx", "Access Control", ["eight hours", "8 hours"]),
        ("Data in transit TLS", "What version of Transport Layer Security is enforced for customer data?", "technova_security_policy.docx", "Data Encryption", ["TLS 1.2"]),
        ("Data at rest AES", "How is stored customer information encrypted at rest?", "technova_security_policy.docx", "Data Encryption", ["AES-256"]),
        ("Screen lock timer", "How fast does an unattended company MacBook lock its display?", "technova_security_policy.docx", "Device Security", ["five minutes", "5 minutes"]),
        ("Breach notification timeline", "How quickly must European authorities be notified of a customer data breach?", "technova_security_policy.docx", "Incident Response", ["72 hours", "GDPR"]),
        ("SOC 2 Type 2 report", "Does TechNova possess an active SOC 2 Type II audit report?", "technova_security_policy.docx", "Compliance and Certification", ["SOC 2 Type II"]),
        ("Simulated phishing policy", "What is the penalty if an employee clicks on a simulated phishing test?", "technova_security_policy.docx", "Security Training", ["refresher", "not penalised"]),
        # Additional out-of-scope variations
        ("Tuition reimbursement degree", "Does the company pay for a Master's degree tuition in full?", None, None, ["not", "no information"]),
        ("Paternity leave for contractors", "Are independent contractors eligible for paid parental leave?", "technova_employee_handbook.pdf", "Welcome to TechNova", ["Contractors", "individual agreements"]),
        ("Office pet policy", "Can employees bring emotional support animals to the Dublin office?", None, None, ["not", "does not contain"]),
        ("Company equity grant", "How many stock units are granted to Senior Engineers on hire?", None, None, ["not", "no information"])
    ]

    for item in VARIATIONS:
        q, exp_q, doc, sec, keys = item
        cases.append({
            "id": f"eval_{case_counter:03d}",
            "question": exp_q,
            "expected_document": doc,
            "expected_section": sec,
            "expected_key_phrases": keys,
            "ground_truth_answer": f"Answer grounded in {doc or 'external / unmentioned'}."
        })
        case_counter += 1

    # Fill remaining cases up to 100 with realistic cross-document queries
    REMAINING_FILLERS = [
        ("What are the core hours and what happens if a meeting is held outside them?", "technova_employee_handbook.pdf", "Working Hours", ["10:00 to 16:00", "recorded"]),
        ("If I get sick for 4 days in a row, do I need a doctor note?", "technova_employee_handbook.pdf", "Sick Leave", ["medical certificate", "three consecutive"]),
        ("Who leads the response when a security incident occurs?", "technova_security_policy.docx", "Incident Response", ["Security team"]),
        ("What is the penalty for violating the Information Security Policy?", "technova_security_policy.docx", "Purpose and Scope", ["termination", "disciplinary"]),
        ("How many pipelines can I create on the Enterprise tier?", "technova_product_guide.pdf", "Pricing Tiers", ["unlimited pipelines"]),
        ("Can private connectors be shared publicly on the NovaCloud marketplace?", "technova_product_guide.pdf", "Connectors", ["visible only", "private"]),
        ("Are sub-processors listed publicly?", "technova_security_policy.docx", "Vendor Security", ["sub-processors", "trust page", "30 days"]),
        ("How are encryption keys managed and when are they rotated?", "technova_security_policy.docx", "Data Encryption", ["key management service", "annually"]),
        ("What is the rate limit for the REST API in requests per minute?", "technova_product_guide.pdf", "Platform Limits", ["600 requests per minute"]),
        ("What happens to employee access rights when their employment ends?", "technova_security_policy.docx", "Access Control", ["two hours", "disabled"]),
        ("What is the maximum size of a file in a single step?", "technova_product_guide.pdf", "Platform Limits", ["5 GB"]),
        ("How many weeks of leave does a secondary caregiver receive?", "technova_employee_handbook.pdf", "Parental Leave", ["6 weeks"]),
        ("How are expense claims submitted and when must receipts be turned in?", "technova_employee_handbook.pdf", "Learning Budget", ["30 days", "Expenses portal"])
    ]

    idx = 0
    while len(cases) < 100:
        item = REMAINING_FILLERS[idx % len(REMAINING_FILLERS)]
        cases.append({
            "id": f"eval_{case_counter:03d}",
            "question": f"Query variation #{case_counter}: {item[0]}",
            "expected_document": item[1],
            "expected_section": item[2],
            "expected_key_phrases": item[3],
            "ground_truth_answer": f"Grounded fact verification for {item[1]}."
        })
        case_counter += 1
        idx += 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(f"Generated {len(cases)} comprehensive evaluation test cases at: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_100_eval_cases()

