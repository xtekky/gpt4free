from urllib.parse import urljoin


def issues_url(organization, repository):
    return urljoin(
        "https://github.com/", f"{organization}/{repository}/issues/",
    )


ISSUES_URL = issues_url("python-jsonschema", "jsonschema")
TEST_SUITE_ISSUES_URL = issues_url("json-schema-org", "JSON-Schema-Test-Suite")


def bug(issue=None):
    message = "A known bug."
    if issue is not None:
        message += f" See {urljoin(ISSUES_URL, str(issue))}."
    return message


def test_suite_bug(issue):
    return (
        "A known test suite bug. "
        f"See {urljoin(TEST_SUITE_ISSUES_URL, str(issue))}."
    )
