function showLoading() {
    // Hide the form and the submit button
    const form = document.getElementById('lesson-form');
    // We don't want to hide the whole form normally, just dim it or hide the button,
    // but hiding the button and showing the spinner is standard.
    const submitBtn = document.getElementById('submit-btn');
    const loadingState = document.getElementById('loading-state');

    if(submitBtn && loadingState) {
        submitBtn.style.display = 'none';
        loadingState.classList.remove('hidden');
    }
}
