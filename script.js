// Elements
const loginContainer = document.getElementById("login-container");
const signupContainer = document.getElementById("signup-container");
const dashboardContainer = document.getElementById("dashboard-container");

const showSignup = document.getElementById("show-signup");
const showLogin = document.getElementById("show-login");

const loginForm = document.getElementById("login-form");
const signupForm = document.getElementById("signup-form");

const welcomeMessage = document.getElementById("welcome-message");

const profileSection = document.getElementById("profile-section");
const howtoSection = document.getElementById("howto-section");
const aboutSection = document.getElementById("about-section");
const trainSection = document.getElementById("train-section");

const profileBtn = document.getElementById("menu-profile");
const howtoBtn = document.getElementById("menu-howto");
const aboutBtn = document.getElementById("menu-about");
const logoutBtn = document.getElementById("menu-logout");

const trainBtn = document.getElementById("btn-train-model");
const trainForm = document.getElementById("train-form");
const metricsOutput = document.getElementById("metrics-output");

let currentUser = {};
let metricsChart = null;

// Show/hide login/signup
showSignup.addEventListener("click", () => {
  loginContainer.classList.add("hidden");
  signupContainer.classList.remove("hidden");
});
showLogin.addEventListener("click", () => {
  signupContainer.classList.add("hidden");
  loginContainer.classList.remove("hidden");
});

// Signup
signupForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const first = document.getElementById("signup-first").value;
  const last = document.getElementById("signup-last").value;
  const email = document.getElementById("signup-email").value;
  const password = document.getElementById("signup-password").value;
  const confirm = document.getElementById("signup-confirm-password").value;
  if(password !== confirm){ alert("Passwords do not match"); return; }
  currentUser = { first, last, email };
  showDashboard();
});

// Login
loginForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const email = document.getElementById("login-email").value;
  currentUser = { first: email.split('@')[0], last: "", email };
  showDashboard();
});

// Show dashboard
function showDashboard(){
  loginContainer.classList.add("hidden");
  signupContainer.classList.add("hidden");
  dashboardContainer.classList.remove("hidden");
  typeWelcomeMessage(`Hi ${currentUser.first}, I am the Teachable Machine.`);
  profileSection.classList.add("hidden");
  howtoSection.classList.add("hidden");
  aboutSection.classList.add("hidden");
  trainSection.classList.add("hidden");
}

// Animated welcome message
function typeWelcomeMessage(message){
    welcomeMessage.textContent = "";
    let i = 0;
    const speed = 50; // milliseconds per character
    function typeChar(){
        if(i < message.length){
            welcomeMessage.textContent += message.charAt(i);
            i++;
            setTimeout(typeChar, speed);
        }
    }
    typeChar();
}

// Menu buttons
profileBtn.addEventListener("click", ()=>{
  profileSection.innerHTML = `<h2>Profile</h2><p>First Name: ${currentUser.first}<br>Last Name: ${currentUser.last}<br>Email: ${currentUser.email}</p>`;
  profileSection.classList.remove("hidden");
  howtoSection.classList.add("hidden");
  aboutSection.classList.add("hidden");
  trainSection.classList.add("hidden");
});
howtoBtn.addEventListener("click", ()=>{
  howtoSection.innerHTML = `<h2>How to Use</h2>
  <ol>
    <li>Login or sign up to access the dashboard.</li>
    <li>Click "Train ML Model" to upload your CSV.</li>
    <li>Select the ML task and provide the target column (if needed).</li>
    <li>Click "Train" to run the model and view metrics dynamically.</li>
    <li>View trained models and metrics from your profile section.</li>
  </ol>`;
  howtoSection.classList.remove("hidden");
  profileSection.classList.add("hidden");
  aboutSection.classList.add("hidden");
  trainSection.classList.add("hidden");
});
aboutBtn.addEventListener("click", ()=>{
  aboutSection.innerHTML = `<h2>About</h2>
  <p>The Teachable Machine allows you to train AI/ML models interactively using your own data. 
  Supported tasks: Classification, Regression, Clustering, and Anomaly Detection. 
  Models are trained on the backend (FastAPI) and metrics are displayed dynamically on the frontend using Chart.js. 
  This application supports incremental learning, allows saving models per user, and provides step-by-step guidance.</p>`;
  aboutSection.classList.remove("hidden");
  profileSection.classList.add("hidden");
  howtoSection.classList.add("hidden");
  trainSection.classList.add("hidden");
});

// Log out
logoutBtn.addEventListener("click", ()=>{
  currentUser = {};
  dashboardContainer.classList.add("hidden");
  loginContainer.classList.remove("hidden");
});

// Show Train ML page
trainBtn.addEventListener("click", ()=>{
  trainSection.classList.remove("hidden");
  profileSection.classList.add("hidden");
  howtoSection.classList.add("hidden");
  aboutSection.classList.add("hidden");
});

// Train ML form
trainForm.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const file = document.getElementById("csv-file").files[0];
  const task = document.getElementById("task-select").value;
  const target_column = document.getElementById("target-column").value;

  if(!file || !task){ alert("Please upload CSV and select a task"); return; }

  metricsOutput.textContent = "Training in progress...";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("task", task);
  if(target_column) formData.append("target_column", target_column);

  try{
    const res = await fetch(`http://127.0.0.1:8000/train/${currentUser.first}`, { method: "POST", body: formData });
    if(!res.ok) throw new Error("Training failed");
    const data = await res.json();
    metricsOutput.textContent = JSON.stringify(data.metrics, null, 2);
    renderChart(data.metrics);
  }catch(err){
    metricsOutput.textContent = "Error: " + err.message;
  }
});

// Chart.js
function renderChart(metrics){
  const ctx = document.getElementById("metrics-chart").getContext("2d");
  if(metricsChart) metricsChart.destroy();

  metricsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Object.keys(metrics),
      datasets: [{
        label: 'Metrics',
        data: Object.values(metrics),
        backgroundColor: Object.keys(metrics).map((_, i) => `hsl(${i*60}, 70%, 50%)`),
        borderColor: '#fff',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      animation: {
        duration: 1000,
        easing: 'easeOutBounce'
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#333',
          titleColor: '#fff',
          bodyColor: '#fff'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { color: '#fff' }
        },
        x: { ticks: { color: '#fff' } }
      }
    }
  });
}
