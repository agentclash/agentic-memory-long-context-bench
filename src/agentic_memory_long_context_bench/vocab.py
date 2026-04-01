FIRST_NAMES = [
    "Priya",
    "Ava",
    "Noah",
    "Maya",
    "Liam",
    "Ivy",
    "Rohan",
    "Anika",
]

PLANS = ["starter", "pro", "enterprise", "team"]
TIMEZONES = ["UTC+5:30", "UTC-8", "UTC+1", "UTC+9"]
THEMES = ["dark", "light"]
DEVICES = ["MacBook Pro", "ThinkPad X1", "Dell XPS 13", "Framework Laptop"]
CHANNELS = ["email", "slack", "sms"]

ISSUES = [
    "a login loop after SSO rotation",
    "webhook delivery failures after a billing deployment",
    "a dashboard permission mismatch after a role change",
    "duplicate invoice emails after a subscription migration",
]

ATTEMPTED_STEPS = [
    "clearing browser cookies",
    "resetting the password",
    "flushing DNS",
    "restarting the router",
    "replaying one failed webhook",
    "checking the role mapping",
]

DISTRACTOR_MESSAGES = [
    "I have a team offsite next week.",
    "The dashboard color palette looked different yesterday.",
    "We migrated some unrelated reports last month.",
    "My colleague also asked about the admin console.",
    "The release notes mentioned analytics changes.",
    "I was reviewing quarterly planning docs earlier.",
]

PROCEDURES = {
    "login_loop": {
        "content": "Troubleshoot a login loop after an SSO or cookie mismatch.",
        "steps": [
            "Confirm whether the loop started after an SSO or cookie change",
            "Clear cookies for the product domain",
            "Retry sign-in in an incognito window",
            "Re-authenticate the SSO session",
        ],
        "must_include": ["clear cookies", "incognito", "re-authenticate"],
    },
    "webhook_failures": {
        "content": "Troubleshoot webhook delivery failures after a billing deployment.",
        "steps": [
            "Check the deployment timestamp against the first failed webhook",
            "Inspect signature verification settings",
            "Replay one failed event in a staging or dry-run environment",
            "Escalate with request ids if signature replay still fails",
        ],
        "must_include": ["deployment timestamp", "signature", "replay"],
    },
}
