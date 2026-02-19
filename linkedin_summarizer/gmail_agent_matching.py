"""
Enhanced Gmail Agent with Job Matching
- Extracts jobs from emails
- Visits LinkedIn job pages
- Rates each job against your profile
"""

import os
import base64
import re
from typing import List, Dict
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_ollama import OllamaLLM

# For web scraping
import requests
from bs4 import BeautifulSoup

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def extract_jobs_from_linkedin_text(text: str) -> List[Dict]:
    """Extract jobs from LinkedIn email format."""
    
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
        
        # Found a job URL
        if 'View job:' in line or 'linkedin.com/comm/jobs/view' in line:
            
            # Extract clean URL
            url_match = re.search(r'https://www\.linkedin\.com/comm/jobs/view/(\d+)', line)
            job_url = url_match.group(0) if url_match else ''
            
            # Work backwards to find job title, company, location
            title_idx = None
            company_idx = None
            location_idx = None
            
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
                    ', ' in lines[i - offset]
                ):
                    location_idx = i - offset
                elif company_idx is None:
                    company_idx = i - offset
                elif title_idx is None:
                    title_idx = i - offset
                    break
            
            # If we found at least title and company
            if title_idx and company_idx:
                job = {
                    'title': lines[title_idx],
                    'company': lines[company_idx],
                    'location': lines[location_idx] if location_idx else 'Not specified',
                    'url': job_url,
                    'salary': ''
                }
                
                # Look for salary
                if location_idx:
                    for check_idx in range(location_idx + 1, i):
                        if '$' in lines[check_idx] or '/year' in lines[check_idx]:
                            job['salary'] = lines[check_idx]
                            break
                
                jobs.append(job)
        
        i += 1
    
    return jobs


class LinkedInJobScraper:
    """
    Scrapes LinkedIn job pages (without authentication).
    
    Note: LinkedIn blocks most scraping. This is a simple version
    that works for publicly accessible job posts. For production,
    you'd need LinkedIn API access or a proper scraper service.
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
    
    def get_job_description(self, url: str) -> str:
        """
        Attempt to scrape job description from LinkedIn.
        
        WARNING: LinkedIn actively blocks scrapers. This may not work
        reliably. Consider alternatives:
        - LinkedIn API (requires approval)
        - Manual copy-paste of job descriptions
        - Third-party job aggregators
        """
        
        if not url:
            return "No URL provided"
        
        try:
            # Add delay to be respectful
            time.sleep(2)
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return f"Could not access (status {response.status_code})"
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find job description
            # LinkedIn's HTML structure changes frequently, so this is fragile
            desc_div = soup.find('div', {'class': 'description__text'})
            if desc_div:
                return desc_div.get_text(strip=True, separator='\n')
            
            # Alternative: look for any div with "description" in class
            for div in soup.find_all('div'):
                if div.get('class') and any('description' in c.lower() for c in div.get('class')):
                    text = div.get_text(strip=True, separator='\n')
                    if len(text) > 100:  # Likely the description
                        return text
            
            return "Could not extract description (LinkedIn may have blocked scraping)"
        
        except Exception as e:
            return f"Error scraping: {str(e)}"


class JobMatcher:
    """
    Matches jobs against your profile using LLM.
    """
    
    def __init__(self, resume_text: str):
        self.resume_text = resume_text
        self.llm = OllamaLLM(model="mistral", temperature=0.2)
    
    def rate_job_match(self, job: Dict, job_description: str) -> Dict:
        """
        Rate how well a job matches your profile.
        
        Returns:
            {
                'score': int (1-10),
                'reasoning': str,
                'strengths': list,
                'gaps': list
            }
        """
        
        # If we couldn't get the description, can't rate accurately
        if 'Could not' in job_description or 'Error' in job_description:
            return {
                'score': 0,
                'reasoning': 'Could not access job description',
                'strengths': [],
                'gaps': []
            }
        
        prompt = f"""You are a job matching expert. Rate how well this job matches the candidate's profile.

CANDIDATE PROFILE:
{self.resume_text[:2000]}

JOB:
Title: {job['title']}
Company: {job['company']}
Location: {job['location']}

JOB DESCRIPTION:
{job_description[:3000]}

Analyze the match and provide:
1. Match score (1-10, where 10 is perfect match)
2. Key strengths (what makes this a good match)
3. Potential gaps (skills or experience the candidate might lack)
4. Brief reasoning

Format your response EXACTLY as:
SCORE: [number 1-10]
STRENGTHS:
- [strength 1]
- [strength 2]
- [strength 3]
GAPS:
- [gap 1]
- [gap 2]
REASONING: [1-2 sentence explanation of the score]
"""
        
        try:
            response = self.llm.invoke(prompt)
            
            # Parse the response
            score_match = re.search(r'SCORE:\s*(\d+)', response)
            score = int(score_match.group(1)) if score_match else 5
            
            # Extract strengths
            strengths = []
            strengths_section = re.search(r'STRENGTHS:(.*?)(?:GAPS:|REASONING:|$)', response, re.DOTALL)
            if strengths_section:
                strengths = [
                    line.strip('- ').strip() 
                    for line in strengths_section.group(1).split('\n') 
                    if line.strip().startswith('-')
                ]
            
            # Extract gaps
            gaps = []
            gaps_section = re.search(r'GAPS:(.*?)(?:REASONING:|$)', response, re.DOTALL)
            if gaps_section:
                gaps = [
                    line.strip('- ').strip() 
                    for line in gaps_section.group(1).split('\n') 
                    if line.strip().startswith('-')
                ]
            
            # Extract reasoning
            reasoning_match = re.search(r'REASONING:\s*(.+)', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response[:200]
            
            return {
                'score': score,
                'reasoning': reasoning,
                'strengths': strengths[:3],  # Top 3
                'gaps': gaps[:2]  # Top 2
            }
        
        except Exception as e:
            return {
                'score': 0,
                'reasoning': f'Error during matching: {str(e)}',
                'strengths': [],
                'gaps': []
            }


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


class EnhancedLinkedInEmailAgent:
    """
    Enhanced agent with job matching capability.
    """
    
    def __init__(self, credentials_file='credentials.json', resume_file=None):
        self.gmail = GmailConnector(credentials_file)
        self.scraper = LinkedInJobScraper()
        
        # Load resume
        if resume_file and os.path.exists(resume_file):
            with open(resume_file, 'r') as f:
                resume_text = f.read()
        else:
            # Default resume summary (you can replace this)
            resume_text = """
            PhD in Computer Science with 10+ years experience in:
            - Kubernetes, AWS, distributed systems
            - Platform engineering, infrastructure at scale
            - Deep learning, PyTorch, reinforcement learning
            - CI/CD, MLOps, cloud-native architectures
            - Published researcher (4 papers, 1 US patent)
            
            Target roles: ML Platform Engineer, Senior Platform Engineer,
            Infrastructure Engineer at AI companies
            """
        
        self.matcher = JobMatcher(resume_text)
        print("✅ Enhanced LinkedIn Email Agent initialized")
    
    def process_linkedin_emails_with_matching(
        self,
        max_results=5,
        max_jobs_to_rate=3,
        min_score_threshold=6
    ):
        """
        Process emails, extract jobs, rate matches.
        
        Args:
            max_results: Max emails to process
            max_jobs_to_rate: Max jobs to rate per email (to avoid rate limits)
            min_score_threshold: Only show jobs with score >= this
        """
        
        emails = self.gmail.get_linkedin_emails(max_results)
        
        all_matches = []
        
        for email in emails:
            print(f"\nProcessing: {email['subject'][:60]}...")
            
            # Extract jobs
            jobs = extract_jobs_from_linkedin_text(email['body_plain'])
            
            if not jobs:
                print("  No jobs found")
                continue
            
            print(f"  Found {len(jobs)} jobs, rating top {min(max_jobs_to_rate, len(jobs))}...")
            
            # Rate top N jobs
            for i, job in enumerate(jobs[:max_jobs_to_rate]):
                print(f"    Rating job {i+1}/{min(max_jobs_to_rate, len(jobs))}: {job['title'][:40]}...")
                
                # Get job description
                description = self.scraper.get_job_description(job['url'])
                
                # Rate the match
                match_result = self.matcher.rate_job_match(job, description)
                
                # Add to results
                if match_result['score'] >= min_score_threshold:
                    all_matches.append({
                        'job': job,
                        'match': match_result,
                        'email_date': email['date']
                    })
                    print(f"      ✅ Score: {match_result['score']}/10")
                else:
                    print(f"      ⊘ Score: {match_result['score']}/10 (below threshold)")
        
        # Sort by score
        all_matches.sort(key=lambda x: x['match']['score'], reverse=True)
        
        return all_matches


def main():
    """Demo the enhanced agent."""
    
    print("="*70)
    print("ENHANCED LINKEDIN EMAIL AGENT WITH JOB MATCHING")
    print("="*70)
    
    if not os.path.exists('credentials.json'):
        print("\n❌ credentials.json not found!")
        return
    
    try:
        # Initialize agent
        # You can pass resume_file='resume.txt' to load your actual resume
        agent = EnhancedLinkedInEmailAgent()
        
        print("\n⚠️  NOTE: LinkedIn blocks most scraping.")
        print("If job descriptions can't be fetched, the agent will note it.")
        print("For production, consider LinkedIn API or manual input.\n")
        
        # Process emails and rate jobs
        matches = agent.process_linkedin_emails_with_matching(
            max_results=2,  # Only 2 emails for demo
            max_jobs_to_rate=2,  # Only top 2 jobs per email
            min_score_threshold=6  # Only show 6+ matches
        )
        
        # Display results
        print("\n" + "="*70)
        print("TOP MATCHES (sorted by score)")
        print("="*70)
        
        if not matches:
            print("\nNo jobs met the minimum score threshold.")
        
        for i, match in enumerate(matches, 1):
            job = match['job']
            rating = match['match']
            
            print(f"\n{i}. {job['title']}")
            print(f"   Company: {job['company']}")
            print(f"   Location: {job['location']}")
            print(f"   Match Score: {rating['score']}/10")
            print(f"   Reasoning: {rating['reasoning']}")
            
            if rating['strengths']:
                print(f"   Strengths:")
                for s in rating['strengths']:
                    print(f"     + {s}")
            
            if rating['gaps']:
                print(f"   Gaps:")
                for g in rating['gaps']:
                    print(f"     - {g}")
            
            print(f"   URL: {job['url']}")
            print("-" * 70)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
