numgrades=0;
class_exists= false;
var students_list = [];
var numfacesdb = 0;

function startTime() {
    const today = new Date(); // Extract the full date
    const formattedDate = today.toISOString().slice(0, 10); // Extract the date (YYYY-MM-DD)
    const time = today.toTimeString().slice(0, 8); // Extract the time (HH:MM:SS)
    document.getElementById('time').innerHTML = `D: ${formattedDate} T: ${time}`;  // Combine date and time
    setTimeout(startTime, 1000);
}
function open_add_face() {
    document.getElementById("mysidenav").style.width = "25vw";
    console.log(444)
    console.log(attendance_mode1)
  }
function open_remove_face() {
    console.log(3)
    document.getElementById("mysidenav2").style.width = "25vw";
  }
function attendance() {
    console.log("External JavaScript function triggered!");
    document.getElementById('nttype').innerHTML = "Attendance Mode";
    document.getElementById("mysidenav3").style.width = "25vw";
    
  }

function facesindb() {
  document.getElementById("mysidenav4").style.width = "25vw";
  var box = document.getElementById("facesdblistid");
  console.log(facesdblist[0])
  for (i=numfacesdb; i<facesdblist.length; i++)
  {
    console.log(i)
    var div = document.createElement("div");
    div.textContent = facesdblist[i];
    box.appendChild(div);
  }
  numfacesdb = facesdblist.length;
}
var numnames=0
var nameslist2 = [];
var facesdblist = [];
function datarefresh() {
    axios.get("/dataupdate")
    .then(function(response) {
        var which_page = response.data.which_page;
      if (which_page=="main"){
          var nameslist = response.data.identnames;
          var nameslist2 = response.data.identnames;
          var container = document.getElementById("name_list1");
          facesdblist = response.data.facesdb;
          for (i = numnames; i<nameslist.length; i++)
          {
            console.log(nameslist2)
            console.log(nameslist2.length)

            var div = document.createElement("div");
            div.textContent = nameslist[i] + "'s Face Was Detected";
            container.appendChild(div);
            let listbox = document.querySelector('.loglist');
            listbox.scrollTop = -listbox.scrollHeight;
            if (nameslist2.length<14)
              document.querySelector('.loglist').style.height = (nameslist2.length * 5.2) + 'vh';
            else
              document.querySelector('.loglist').style.height =  '75vh';
          }
          numnames=nameslist.length
      }
      if (which_page=="face_lists"){
          var grade_names_list = response.data.grade_names_list;
          var class_names_list = response.data.class_names_list;
          students_list = response.data.students_list;
          var grade_id_list = response.data.grade_id_list;
          var class_id_list = response.data.class_id_list;
          var class_grade_id_list = response.data.class_grade_id_list;
          var verticalBar = document.querySelector(".vertical-bar");
          var dropdownContainer = verticalBar.querySelector(".dropdown-container");
          var newgradesContainer = dropdownContainer.querySelector(".new-classes-here");
          for(i = numgrades; i<grade_names_list.length; i++){
              console.log("loop here")
              var gradeButton = document.createElement("button");
              gradeButton.className = "dropdown-btn";
              gradeButton.innerHTML = `${grade_names_list[i]}th grade <i class="fa fa-caret-down"></i>`;
              gradeButton.setAttribute("onclick", `showrow('${i+1}')`);
              var gradeDropdownContainer = document.createElement("div");
              gradeDropdownContainer.className = "dropdown-container";
              for (g = 0; g<class_grade_id_list.length; g++){
                  if (class_grade_id_list[g]==grade_id_list[i]){
                      class_exists = true
                      var classButton = document.createElement("button");
                      classButton.className = "dropdown-btn";
                      classButton.innerHTML = `Class ${class_names_list[g]}`;
                      classButton.setAttribute("onclick", `showClass('${class_names_list[g]}',${class_id_list[g]})`);
                      gradeDropdownContainer.appendChild(classButton);
                  }
              }
              if (class_exists == false){
                  var classButton = document.createElement("button");
                  classButton.className = "dropdown-btn";
                  classButton.innerHTML = "No classes exist";
                  gradeDropdownContainer.appendChild(classButton);
              }
              class_exists = false
              // var classButton = document.createElement("button");
              // classButton.className = "dropdown-btn";
              // classButton.innerHTML = "Class 1";
              // classButton.setAttribute("onclick", "showClass('Class 1')");

              // Create the link for adding a grade
              var addGradeLink = document.createElement("a");
              addGradeLink.className = "button";
              addGradeLink.setAttribute("onclick", `addclass(${i})`);
              addGradeLink.innerText = "Add Class";
              var formDiv = document.createElement("div");
              formDiv.className = "button";
              formDiv.id = "addsmthgclass";
              var form = document.createElement("form");
              form.setAttribute("action", "/add_class");
              form.setAttribute("method", "POST");
              var hiddenInput = document.createElement("input");
              hiddenInput.type = "hidden";
              hiddenInput.id = "gradesection";
              hiddenInput.name = "gradesection";
              hiddenInput.value = grade_id_list[i];
              var classNameInput = document.createElement("input");
              classNameInput.type = "text";
              classNameInput.id = "numinp";
              classNameInput.name = "classname";
              classNameInput.placeholder = "Name";
              classNameInput.required = true;
              var submitInput = document.createElement("input");
              submitInput.type = "submit";
              submitInput.id = "subinp";
              submitInput.value = "Add Class";


              var addGradeLinkr = document.createElement("a");
              addGradeLinkr.className = "button";
              addGradeLinkr.setAttribute("onclick", `removeclass(${i})`);
              addGradeLinkr.innerText = "Remove Class";
              var formDivr = document.createElement("div");
              formDivr.className = "button";
              formDivr.id = "removesmthgclass";
              var formr = document.createElement("form");
              formr.setAttribute("action", "/remove_class");
              formr.setAttribute("method", "POST");
              var hiddenInputr = document.createElement("input");
              hiddenInputr.type = "hidden";
              hiddenInputr.id = "gradesection";
              hiddenInputr.name = "gradesection";
              hiddenInputr.value = grade_id_list[i];
              var classNameInputr = document.createElement("input");
              classNameInputr.type = "text";
              classNameInputr.id = "numinp";
              classNameInputr.name = "classname";
              classNameInputr.placeholder = "Name";
              classNameInputr.required = true;
              var submitInputr = document.createElement("input");
              submitInputr.type = "submit";
              submitInputr.id = "subinp";
              submitInputr.value = "Remove Class";

              // Append inputs to the form
              form.appendChild(hiddenInput);
              form.appendChild(classNameInput);
              form.appendChild(submitInput);

              formr.appendChild(hiddenInputr);
              formr.appendChild(classNameInputr);
              formr.appendChild(submitInputr);
              // Append form to form div
              formDiv.appendChild(form);
              formDivr.appendChild(formr);
              // Append all elements to the grade dropdown container
              gradeDropdownContainer.appendChild(classButton);
              gradeDropdownContainer.appendChild(addGradeLink);
              gradeDropdownContainer.appendChild(formDiv);
              gradeDropdownContainer.appendChild(addGradeLinkr);
              gradeDropdownContainer.appendChild(formDivr);

              // Append the grade button and its dropdown to the main dropdown container
              newgradesContainer.appendChild(gradeButton);
              newgradesContainer.appendChild(gradeDropdownContainer);
          }


          numgrades= grade_names_list.length;
      }
    })
    .catch(function(error) {console.log(error);});
}
// calls the function daarefresh() every 2 seconds
var intervalID = window.setInterval(datarefresh, 3000);


function close_side() {
    console.log(1)
    document.getElementById("mysidenav").style.width = "0";
    document.getElementById("mysidenav2").style.width = "0";
    document.getElementById("mysidenav3").style.width = "0";
    document.getElementById("mysidenav4").style.width = "0";
    document.getElementById('nttype').innerHTML = "Face Detection Log";
  }

  function downloadnamesnp() {
    //using let so vars don't save
    let content = nameslist2.join("\n");  // makes arr one text var 
    let blob = new Blob([content], { type: "text/plain" }); // turns var into blob inorder to download
    let a = document.createElement("a"); // makes an invisable link yaani <a>
    a.href = URL.createObjectURL(blob); 
    a.download = "Attendance_list_np.txt";  // File name
    document.body.appendChild(a); // adds link to page 
    a.click(); // clicks link
    document.body.removeChild(a); // removes link from page
  }
  function downloadnamesx() {
    let content = nameslist2.join("\n");
    let blob = new Blob([content], { type: "text/csv" });
    let a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "Attendance_list_ex.csv"; 
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function showClass(className, classid) {
    document.getElementById('classnumber').value = classid;
    document.getElementById('classnumber2').value = classid;
    document.getElementById('class-title').innerText = className;
    const studentBody = document.getElementById('student-body');
    studentBody.innerHTML = ''; // Clear previous data
    for (i = 0; i<students_list.length; i++){
        if (students_list[i][4] == classid){
            const row = document.createElement('tr');
            row.innerHTML = `<td>${students_list[i][1]}</td>
                 <td>${students_list[i][2]}</td>
                 <td>${students_list[i][3] == 0 ? 'False' : 'True'}</td>`;
            studentBody.appendChild(row);
        }
    }
    document.querySelectorAll(".classadd")[0].style.display = 'inline';
    document.querySelectorAll(".classadd")[1].style.display = 'inline';
    document.getElementById('student-table').style.display = 'table';
    document.getElementById('buttonsforstudents').style.display = 'flex';
}

function showrow(which) {
    var dropdownc = document.querySelectorAll(".dropdown-container")[which];
    if (dropdownc.style.display === "block") {
        dropdownc.style.display = "none";
    } else {
        dropdownc.style.display = "block";
    }
}
function addclass(num) {
    var dropdownc = document.querySelectorAll("#addsmthgclass")[num];
    if (dropdownc.style.display === "block") {
        dropdownc.style.display = "none";
    } else {
        dropdownc.style.display = "block";
    }
}
function removeclass(num) {
    var dropdownc = document.querySelectorAll("#removesmthgclass")[num];
    if (dropdownc.style.display === "block") {
        dropdownc.style.display = "none";
    } else {
        dropdownc.style.display = "block";
    }
}
function addgrade(num) {
    var dropdownc = document.querySelectorAll("#addsmthggrade")[num];
    if (dropdownc.style.display === "block") {
        dropdownc.style.display = "none";
    } else {
        dropdownc.style.display = "block";
    }
}
function addstudent(num) {
    var dropdownc = document.querySelectorAll("#addsmthgstudent")[num];
    if (dropdownc.style.display === "block") {
        dropdownc.style.display = "none";
    } else {
        dropdownc.style.display = "block";
    }
}

function downloadstudentnames() {
    var namesarray = []
    for (i=0;i<students_list.length;i++){
        namesarray.push(students_list[i][1]);
    }
    //using let so vars don't save
    let content = namesarray.join("\n");  // makes arr one text var 
    let blob = new Blob([content], { type: "text/plain" }); // turns var into blob inorder to download
    let a = document.createElement("a"); // makes an invisable link yaani <a>
    a.href = URL.createObjectURL(blob); 
    a.download = "Attendance_list_np.txt";  // File name
    document.body.appendChild(a); // adds link to page 
    a.click(); // clicks link
    document.body.removeChild(a); // removes link from page
  }
  function downloadstudentdataexl() {
    var namesarray = []
    for (i = 0; i<students_list.length;i++){
        namesarray.push([students_list[i][0], students_list[i][1], students_list[i][2], students_list[i][3]]);
    }
    let csvContent = "Number, Name, ID Number, Face Uploaded";
    namesarray.forEach(function(rowArray) {
        let row = rowArray.join(",");
        csvContent += row + "\n"; 
    });
    let blob = new Blob([csvContent], { type: "text/csv" });
    let a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "Attendance_list_ex.csv"; 
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
