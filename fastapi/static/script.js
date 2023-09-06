async function authenticate() {
    const form = document.getElementById("uploadForm");
    const formData = new FormData(form);
    const image = formData.get("image");
    const medicineName = formData.get("medicine");
    console.log("Hello")
    console.log("FormData:",formData)
    console.log(image)
    console.log(medicineName)
    // create json object
    const data_1 = {"image": image, "medicine": medicineName}
    
    try {
      const response = await fetch('/authenticate', {
        method: 'POST',
        body: formData,
        headers: {'Accept':'application/json'} 
      });
      
      const data = await response.json();
      console.log("Data:",data)
      
      // Update the result div with the authentication result
      const resultDiv = document.getElementById("result");
      resultDiv.textContent = `Authentication Result: ${data.result}`;
    } catch (error) {
      console.error('Error:', error);
    }
  }
  
  async function identify() {
    const form = document.getElementById("uploadForm");
    const formData = new FormData(form);
    
    try {
      const response = await fetch('/identify', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      // Update the result div with the identification result
      const resultDiv = document.getElementById("result");
      resultDiv.textContent = `Identification Result: ${data.result}`;
    } catch (error) {
      console.error('Error:', error);
    }
  }
  
  async function loadMedicineNames() {
    try {
      const response = await fetch('/medicine_names');
      const data = await response.json();
      
      const medicineSelect = document.getElementById("medicine");
      
      // Populate the dropdown list with medicine names
      data.medicine_names.forEach(name => {
        const option = document.createElement("option");
        option.value = name;
        option.text = name;
        medicineSelect.appendChild(option);
      });
    } catch (error) {
      console.error('Error:', error);
    }
  }
  
  loadMedicineNames();
  