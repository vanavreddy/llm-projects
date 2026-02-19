"""
Gmail Agent - Fixed for Actual LinkedIn Format
No · separator - jobs are on separate lines
"""

import os
import base64
import re
from typing import List, Dict

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_ollama import OllamaLLM

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def extract_jobs_from_linkedin_text(text: str) -> List[Dict]:
    """
    Extract jobs from LinkedIn format:
    
    Job Title
    Company
    Location
    [optional: connections, salary, etc]
    View job: [URL]
    """
    
    jobs = []
    
    # Remove URL encoding
    text = re.sub(r'=([0-9A-F]{2})', lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r'=\r?\n', '', text)
    
    # Split into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Find "View job:" lines as anchors
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Found a "View job:" line - work backwards to find the job
        if 'View job:' in line or (i > 0 and 'linkedin.com/comm/jobs/view' in line):
            
            # Work backwards to find job title, company, location
            # Pattern: Title is 3-4 lines before "View job:"
            
            title_idx = None
            company_idx = None
            location_idx = None
            
            # Look back up to 6 lines
            for offset in range(1, min(7, i + 1)):
                check_line = lines[i - offset].lower()
                
                # Skip noise lines
                if any(skip in check_line for skip in [
                    'view job:', 'connection', 'actively hiring', 
                    'your job alert', 'new jobs', 'match your preferences'
                ]):
                    continue
                
                # Assign positions (reading backwards)
                if location_idx is None and (
                    'United States' in lines[i - offset] or
                    'VA' in lines[i - offset] or
                    'Remote' in lines[i - offset] or
                    ', ' in lines[i - offset]  # "City, State" pattern
                ):
                    location_idx = i - offset
                elif company_idx is None:
                    company_idx = i - offset
                elif title_idx is None:
                    title_idx = i - offset
                    break  # Got all three
            
            # If we found at least title and company
            if title_idx and company_idx:
                job = {
                    'title': lines[title_idx],
                    'company': lines[company_idx],
                    'location': lines[location_idx] if location_idx else 'Not specified',
                    'salary': '',
                    'link': line.replace('View job:', '').strip() if 'View job:' in line else line
                }
                
                # Look for salary between location and View job
                if location_idx:
                    for check_idx in range(location_idx + 1, i):
                        if '$' in lines[check_idx] or '/year' in lines[check_idx]:
                            job['salary'] = lines[check_idx]
                            break
                
                jobs.append(job)
        
        i += 1
    
    return jobs


class GmailConnector:
    """Gmail API connector."""
    
    def __init__(self, credentials_file='credentials.json'):
        self.credentials_file = credentials_file
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        creds = None
        
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(f"{self.credentials_file} not found")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('gmail', 'v1', credentials=creds)
        print("✅ Authenticated with Gmail")
    
    def get_linkedin_emails(self, max_results=10):
        query = 'from:jobalerts-noreply@linkedin.com OR from:jobs-noreply@linkedin.com'
        
        try:
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            print(f"Found {len(messages)} LinkedIn emails")
            
            emails = []
            for msg in messages:
                email_data = self._get_email_details(msg['id'])
                if email_data:
                    emails.append(email_data)
            
            return emails
        
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def _get_email_details(self, msg_id):
        try:
            message = self.service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()
            
            headers = message['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            body_plain, body_html = self._extract_body(message['payload'])
            
            return {
                'id': msg_id,
                'subject': subject,
                'date': date,
                'body_plain': body_plain,
                'body_html': body_html
            }
        
        except Exception as e:
            print(f"Error getting email: {e}")
            return None
    
    def _extract_body(self, payload):
        body_plain = ""
        body_html = ""
        
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                    body_plain = base64.urlsafe_b64decode(
                        part['body']['data']
                    ).decode('utf-8', errors='ignore')
                elif part['mimeType'] == 'text/html' and 'data' in part['body']:
                    body_html = base64.urlsafe_b64decode(
                        part['body']['data']
                    ).decode('utf-8', errors='ignore')
        else:
            if 'data' in payload['body']:
                data = base64.urlsafe_b64decode(
                    payload['body']['data']
                ).decode('utf-8', errors='ignore')
                
                if payload.get('mimeType') == 'text/plain':
                    body_plain = data
                else:
                    body_html = data
        
        return body_plain, body_html


class LinkedInEmailAgent:
    """LinkedIn email processor."""
    
    def __init__(self, credentials_file='credentials.json'):
        self.gmail = GmailConnector(credentials_file)
        self.llm = OllamaLLM(model="mistral", temperature=0.3)
        print("✅ LinkedIn Email Agent initialized")
    
    def process_linkedin_emails(self, max_results=10):
        emails = self.gmail.get_linkedin_emails(max_results)
        
        results = []
        for email in emails:
            print(f"\nProcessing: {email['subject'][:60]}...")
            
            # Classify
            email_type = self._classify(email['subject'], email['body_plain'])
            
            # Extract jobs
            jobs = []
            if 'job' in email_type or 'job' in email['subject'].lower():
                jobs = extract_jobs_from_linkedin_text(email['body_plain'])
                content = self._format_jobs(jobs)
                if jobs:
                    print(f"  ✅ Extracted {len(jobs)} jobs")
                else:
                    print(f"  ⚠️  No jobs extracted")
            else:
                content = "LinkedIn notification"
            
            results.append({
                'date': email['date'],
                'subject': email['subject'],
                'type': 'jobs' if jobs else 'other',
                'content': content,
                'raw_jobs': jobs
            })
        
        return results
    
    def _classify(self, subject, body):
        # Very inclusive - most LinkedIn emails from jobalerts are jobs
        subject_lower = subject.lower()
        
        # If it has quotes around a search term, it's a job alert
        if '"' in subject:
            return 'jobs'
        
        # Other job indicators
        job_keywords = [
            'job alert', 'jobs you may', 'similar to', 'new jobs',
            'recommended for you', 'match your preferences',
            'jobs at', 'jobs for', 'positions at'
        ]
        
        if any(k in subject_lower for k in job_keywords):
            return 'jobs'
        
        # Check if from job alerts email
        # (Already filtered by Gmail query, so assume it's jobs)
        return 'jobs'  # Default to jobs for emails from job alert address
    
    def _format_jobs(self, jobs):
        if not jobs:
            return "No jobs found in email"
        
        output = f"Found {len(jobs)} job(s):\n\n"
        for i, job in enumerate(jobs, 1):
            output += f"{i}. {job['title']}\n"
            output += f"   Company: {job['company']}\n"
            output += f"   Location: {job['location']}\n"
            if job.get('salary'):
                output += f"   Salary: {job['salary']}\n"
            output += "\n"
        
        return output.strip()


def main():
    print("="*70)
    print("LINKEDIN EMAIL AGENT - FIXED FOR REAL FORMAT")
    print("="*70)
    
    if not os.path.exists('credentials.json'):
        print("\n❌ credentials.json not found!")
        return
    
    try:
        agent = LinkedInEmailAgent()
        
        print("\nFetching LinkedIn emails...")
        results = agent.process_linkedin_emails(max_results=5)
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Email {i} ---")
            print(f"Date: {result['date']}")
            print(f"Subject: {result['subject']}")
            print(f"Type: {result['type']}")
            print(f"\nContent:\n{result['content']}")
            print("-" * 70)
        
        print(f"\n✅ Processed {len(results)} emails")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
