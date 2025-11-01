# Contributing to Indian Stock Market Analyzer

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Bugs
- Check if the bug has already been reported in Issues
- Use the bug report template
- Include steps to reproduce, expected behavior, and actual behavior
- Add screenshots if applicable

### Suggesting Enhancements
- Check if the enhancement has already been suggested
- Clearly describe the feature and its benefits
- Explain why this would be useful to most users

### Pull Requests
1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions
- Keep functions focused and small
- Add type hints where appropriate

### Documentation
- Update README.md if adding new features
- Add comments for complex logic
- Update IMPROVEMENTS_SUMMARY.md for major changes
- Keep USAGE_GUIDE.md current

### Testing
- Test your changes thoroughly
- Verify with multiple stocks
- Check different time periods
- Ensure error handling works

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove)
- Keep first line under 50 characters
- Add detailed description if needed

Example:
```
Add sentiment analysis feature

- Integrate news sentiment API
- Add sentiment score to predictions
- Update UI to display sentiment
```

## 🎯 Areas for Contribution

### High Priority
- [ ] Add unit tests (pytest)
- [ ] Implement backtesting framework
- [ ] Add more technical indicators
- [ ] Improve prediction accuracy
- [ ] Add database integration
- [ ] Implement proper logging

### Medium Priority
- [ ] Add more Indian stocks
- [ ] Support for international markets
- [ ] Portfolio optimization features
- [ ] Alert system (email/SMS)
- [ ] Mobile-responsive improvements
- [ ] Dark/light theme toggle

### Low Priority
- [ ] Additional chart types
- [ ] Export to Excel
- [ ] Social media sentiment
- [ ] News integration
- [ ] Cryptocurrency support

## 🚫 What NOT to Contribute

- Changes that make false accuracy claims
- Removal of disclaimers or warnings
- Features that encourage irresponsible trading
- Code without proper error handling
- Undocumented complex features

## 📝 Code Review Process

1. All PRs require review before merging
2. Address review comments promptly
3. Keep PRs focused on single features
4. Ensure CI/CD checks pass
5. Update documentation as needed

## 🔒 Security

- Never commit API keys or secrets
- Use environment variables for sensitive data
- Report security vulnerabilities privately
- Don't include personal data in commits

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License with the same disclaimer as the project.

## 💬 Questions?

- Open an issue for questions
- Tag with "question" label
- Be respectful and patient

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for making this project better! 🚀
