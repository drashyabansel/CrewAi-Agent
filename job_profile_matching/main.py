from job_profile_matching.crew.crew import JobProfilling
from dotenv import load_dotenv
load_dotenv()

def kickoff():
    job_profilling = JobProfilling()

    job_profilling.crew().kickoff(
        inputs={
            'github_url': 'https://github.com/drashyabansel/',
            'job_posting_url': "https://www.naukri.com/job-listings-ai-engineer-trustana-gurugram-1-to-4-years-241224503115?src=seo_srp&sid=17422119983893000&xp=1&px=1"
        }
    )


kickoff()