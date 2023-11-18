import re
from gitlint.rules import CommitRule, RuleViolation
from dataclasses import dataclass


@dataclass
class Directive:
    desc: str
    aliases: list[str] = None
    sub: list[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.sub is None:
            self.sub = []
        self.desc = self._build_description()

    def _build_description(self):
        tab = 4 * ' '
        return (
            self.desc
            + '\n'
            + f'{tab}ALIASES: '
            + ', '.join(self.aliases)
            + '\n'
            + f'{tab}UNDERSCORES: '
            + ', '.join(self.sub)
        )


def get_directives():
    return {
        'BUG': Directive('Found bug', sub=['RESURFACED', 'AGAIN']),
        'FIX': Directive('Fixed bug'),
        'FEATURE': Directive('Added feature'),
        'DEBUG': Directive('Debugging change'),
        'CLEAN': Directive('Cleaned code'),
        'DOCS': Directive('Documentation change'),
        'SAFETY': Directive('Safety enhancement'),
        'REFACTOR': Directive(
            'Refactored code',
            sub=['DOUBT', 'WORKS', 'BREAK', 'CKPT', 'CHECKPOINT'],
        ),
        'TEST': Directive('Test addition/change'),
        'CONFIG': Directive('Config change'),
        'REVERT': Directive('Reverted change'),
        'DEPRECATE': Directive('Deprecated feature/code'),
        'UPDATE': Directive('General update'),
        'PERFORMANCE': Directive('Performance optimization', aliases=['PERF']),
        'DROP': Directive('Dropped code/feature'),
        'REQUEST': Directive('Request-based change'),
        'EMPTY_FOLLOW': Directive(
            'Follow-up to previous commit to provide more info'
        ),
        'REMOVE': Directive('Removed code, feature or file', aliases=['RM']),
        'MERGE': Directive(
            'Commit involving a merge operation', sub=['WAIT', 'READY']
        ),
        'PAUSE': Directive(
            'PAUSE_X Pausing on this branch b/c of X',
            sub=['DOCS', 'FEATURE', 'PERF', 'OVERKILL', 'DESIGN', 'BUG'],
        ),
        'EMPTY': Directive(
            'Empty commit -- should only be used for important notes'
        ),
        'MILESTONE': Directive(
            'Milestone commit -- think as "good, significant checkpoints"'
        ),
        'FALLBACK': Directive(
            'Fallback commit -- think as "milestone set for safety"'
        ),
    }


def generate_directive_names(directives):
    names = []
    for key, directive in directives.items():
        names.append(key)
        names.extend([f"{key}_{sub}" for sub in directive.sub])
        names.extend(directive.aliases)
    return names


class DirectiveCommitRule(CommitRule):
    """Custom gitlint rule to enforce commit message directives."""

    id = "DCR1"
    name = "directive-commit-rule"

    # Define your directives
    directives = get_directives()

    def validate(self, commit):
        directive_names = generate_directive_names(self.directives)
        pattern = r"^(%s): .+" % "|".join(directive_names)

        if not re.match(pattern, commit.message.full):
            violation_msg = (
                f"Commit message '{commit.message.full}' does not follow the"
                " required format. Use one of the following directives: "
                + "\n    "
                + "\n    ".join(directive_names)
            )
            return [RuleViolation(self.id, violation_msg, line_nr=1)]


# if __name__ == "__main__":
#     from gitlint.git import GitCommit

#     commit = GitCommit(
#         'REFACTOR_WO: Refactored and it works\n\nDetailed description.'
#     )
#     rule = DirectiveCommitRule()
#     violations = rule.validate(commit)
#     for violation in violations:
#         print(violation)
