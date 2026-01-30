
$(document).ready(function() {
    // Get the warning message container and the main content container
    const warningContainer = document.querySelector('.warning-container');
    const mainContent = document.getElementById('main-content');
    
    // Get the proceed button
    const proceedBtn = document.getElementById('proceed-btn');
    
    // Add event listener to the proceed button
    proceedBtn.addEventListener('click', function() {
        // Hide the warning message and show the main content
        warningContainer.style.display = 'none';
        mainContent.style.display = 'block';
    });

    $('#characteristic').change(function() {
        var selectedCharacteristic = $(this).val();
        var specificOptionsDropdown = $('#specificOptions');
        
        // Clear previous options
        specificOptionsDropdown.empty();
        
        // Populate options based on the selected characteristic
        if (selectedCharacteristic === 'Age') {
            specificOptionsDropdown.append('<label for="ageOptions">Select Age Range:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="ageOptions">' +
                '<option value="nothing">---</option>' +
                '<option value="21 to 25">21 to 25</option>' +
                '<option value="26 to 30">26 to 30</option>' +
                '<option value="31 and above">31 and above</option>' +
                '<option value="Under 21">Under 21</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="providerCountry">Provider Country:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="providerCountry">' +
                '<option value="nothing">---</option>' +
                '<option value="England">England</option>' +
                '<option value="Northern Ireland">Northern Ireland</option>' +
                '<option value="Scotland">Scotland</option>' +
                '<option value="UK">UK</option>' +
                '<option value="Wales">Wales</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="MoS">Mode of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="MoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All modes">All modes</option>' +
                '<option value="Apprenticeship">Apprenticeship</option>' +
                '<option value="Full-time">Full-time</option>' +
                '<option value="Part-time">Part-time</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="LoS">Level of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="LoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All undergraduates">All undergraduates</option>' +
                '<option value="First degree">First degree</option>' +
                '<option value="Other undergraduate">Other undergraduate</option>' +
                '<option value="Undergraduate with postgraduate component">Undergraduate with postgraduate component</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="Subject">Subject</label>');
            specificOptionsDropdown.append('<select class="form-control" id="Subject">' +
                '<option value="nothing">---</option>' +
                '<option value="Agriculture, food and related studies">Agriculture, food and related studies</option>' +
                '<option value="All subjects">All subjects</option>' +
                '<option value="Architecture, building and planning">Architecture, building and planning</option>' +
                '<option value="Biological and sport sciences">Biological and sport sciences</option>' +
                '<option value="Business and management">Business and management</option>' +
                '<option value="Combined and general studies">Combined and general studies</option>' +
                '<option value="Computing">Computing</option>' +
                '<option value="Design, and creative and performing arts">Design, and creative and performing arts</option>' +
                '<option value="Education and teaching">Education and teaching</option>' +
                '<option value="Engineering and technology">Engineering and technology</option>' +
                '<option value="Geography, earth and environmental studies">Geography, earth and environmental studies</option>' +
                '<option value="Historical, philosophical and religious studies">Historical, philosophical and religious studies</option>' +
                '<option value="Language and area studies">Language and area studies</option>' +
                '<option value="Law">Law</option>' +
                '<option value="Mathematical sciences">Mathematical sciences</option>' +
                '<option value="Media, journalism and communications">Media, journalism and communications</option>' +
                '<option value="Medicine and dentistry">Medicine and dentistry</option>' +
                '<option value="Physical sciences">Physical sciences</option>' +
                '<option value="Psychology">Psychology</option>' +
                '<option value="Social sciences">Social sciences</option>' +
                '<option value="Subjects allied to medicine">Subjects allied to medicine</option>' +
                '<option value="Veterinary sciences">Veterinary sciences</option>' +
                '</select>');

        } else if (selectedCharacteristic === 'Sex') {
            specificOptionsDropdown.append('<label for="sexOptions">Select Sex:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="sexOptions">' +
                '<option value="nothing">---</option>' +
                '<option value="Female">Female</option>' +
                '<option value="Male">Male</option>' +
                '<option value="Other sex">Other sex</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="providerCountry">Provider Country:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="providerCountry">' +
                '<option value="nothing">---</option>' +
                '<option value="England">England</option>' +
                '<option value="Northern Ireland">Northern Ireland</option>' +
                '<option value="Scotland">Scotland</option>' +
                '<option value="UK">UK</option>' +
                '<option value="Wales">Wales</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="MoS">Mode of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="MoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All modes">All modes</option>' +
                '<option value="Apprenticeship">Apprenticeship</option>' +
                '<option value="Full-time">Full-time</option>' +
                '<option value="Part-time">Part-time</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="LoS">Level of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="LoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All undergraduates">All undergraduates</option>' +
                '<option value="First degree">First degree</option>' +
                '<option value="Other undergraduate">Other undergraduate</option>' +
                '<option value="Undergraduate with postgraduate component">Undergraduate with postgraduate component</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="Subject">Subject</label>');
            specificOptionsDropdown.append('<select class="form-control" id="Subject">' +
                '<option value="nothing">---</option>' +
                '<option value="Agriculture, food and related studies">Agriculture, food and related studies</option>' +
                '<option value="All subjects">All subjects</option>' +
                '<option value="Architecture, building and planning">Architecture, building and planning</option>' +
                '<option value="Biological and sport sciences">Biological and sport sciences</option>' +
                '<option value="Business and management">Business and management</option>' +
                '<option value="Combined and general studies">Combined and general studies</option>' +
                '<option value="Computing">Computing</option>' +
                '<option value="Design, and creative and performing arts">Design, and creative and performing arts</option>' +
                '<option value="Education and teaching">Education and teaching</option>' +
                '<option value="Engineering and technology">Engineering and technology</option>' +
                '<option value="Geography, earth and environmental studies">Geography, earth and environmental studies</option>' +
                '<option value="Historical, philosophical and religious studies">Historical, philosophical and religious studies</option>' +
                '<option value="Language and area studies">Language and area studies</option>' +
                '<option value="Law">Law</option>' +
                '<option value="Mathematical sciences">Mathematical sciences</option>' +
                '<option value="Media, journalism and communications">Media, journalism and communications</option>' +
                '<option value="Medicine and dentistry">Medicine and dentistry</option>' +
                '<option value="Physical sciences">Physical sciences</option>' +
                '<option value="Psychology">Psychology</option>' +
                '<option value="Social sciences">Social sciences</option>' +
                '<option value="Subjects allied to medicine">Subjects allied to medicine</option>' +
                '<option value="Veterinary sciences">Veterinary sciences</option>' +
                '</select>');

        } else if (selectedCharacteristic === 'Ethnicity') {
            specificOptionsDropdown.append('<label for="ethnicityOptions">Select Ethnicity:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="ethnicityOptions">' +
                '<option value="nothing">---</option>' +
                '<option value="Asian">Asian</option>' +
                '<option value="Black">Black</option>' +
                '<option value="Mixed">Mixed</option>' +
                '<option value="White">White</option>' +
                '<option value="Other">Other</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="providerCountry">Provider Country:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="providerCountry">' +
                '<option value="nothing">---</option>' +
                '<option value="England">England</option>' +
                '<option value="Northern Ireland">Northern Ireland</option>' +
                '<option value="Scotland">Scotland</option>' +
                '<option value="UK">UK</option>' +
                '<option value="Wales">Wales</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="MoS">Mode of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="MoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All modes">All modes</option>' +
                '<option value="Apprenticeship">Apprenticeship</option>' +
                '<option value="Full-time">Full-time</option>' +
                '<option value="Part-time">Part-time</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="LoS">Level of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="LoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All undergraduates">All undergraduates</option>' +
                '<option value="First degree">First degree</option>' +
                '<option value="Other undergraduate">Other undergraduate</option>' +
                '<option value="Undergraduate with postgraduate component">Undergraduate with postgraduate component</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="Subject">Subject</label>');
            specificOptionsDropdown.append('<select class="form-control" id="Subject">' +
                '<option value="nothing">---</option>' +
                '<option value="Agriculture, food and related studies">Agriculture, food and related studies</option>' +
                '<option value="All subjects">All subjects</option>' +
                '<option value="Architecture, building and planning">Architecture, building and planning</option>' +
                '<option value="Biological and sport sciences">Biological and sport sciences</option>' +
                '<option value="Business and management">Business and management</option>' +
                '<option value="Combined and general studies">Combined and general studies</option>' +
                '<option value="Computing">Computing</option>' +
                '<option value="Design, and creative and performing arts">Design, and creative and performing arts</option>' +
                '<option value="Education and teaching">Education and teaching</option>' +
                '<option value="Engineering and technology">Engineering and technology</option>' +
                '<option value="Geography, earth and environmental studies">Geography, earth and environmental studies</option>' +
                '<option value="Historical, philosophical and religious studies">Historical, philosophical and religious studies</option>' +
                '<option value="Language and area studies">Language and area studies</option>' +
                '<option value="Law">Law</option>' +
                '<option value="Mathematical sciences">Mathematical sciences</option>' +
                '<option value="Media, journalism and communications">Media, journalism and communications</option>' +
                '<option value="Medicine and dentistry">Medicine and dentistry</option>' +
                '<option value="Physical sciences">Physical sciences</option>' +
                '<option value="Psychology">Psychology</option>' +
                '<option value="Social sciences">Social sciences</option>' +
                '<option value="Subjects allied to medicine">Subjects allied to medicine</option>' +
                '<option value="Veterinary sciences">Veterinary sciences</option>' +
                '</select>');
        
        } else if (selectedCharacteristic === 'Domicile') {
            specificOptionsDropdown.append('<label for="domicileOptions">Select Domicile:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="domicileOptions">' +
                '<option value="nothing">---</option>' +
                '<option value="United Kingdom">United Kingdom</option>' +
                '<option value="European Union">European Union</option>' +
                '<option value="Rest of the World">Rest of the World</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="providerCountry">Provider Country:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="providerCountry">' +
                '<option value="nothing">---</option>' +
                '<option value="England">England</option>' +
                '<option value="Northern Ireland">Northern Ireland</option>' +
                '<option value="Scotland">Scotland</option>' +
                '<option value="UK">UK</option>' +
                '<option value="Wales">Wales</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="MoS">Mode of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="MoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All modes">All modes</option>' +
                '<option value="Apprenticeship">Apprenticeship</option>' +
                '<option value="Full-time">Full-time</option>' +
                '<option value="Part-time">Part-time</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="LoS">Level of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="LoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All undergraduates">All undergraduates</option>' +
                '<option value="First degree">First degree</option>' +
                '<option value="Other undergraduate">Other undergraduate</option>' +
                '<option value="Undergraduate with postgraduate component">Undergraduate with postgraduate component</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="Subject">Subject</label>');
            specificOptionsDropdown.append('<select class="form-control" id="Subject">' +
                '<option value="nothing">---</option>' +
                '<option value="Agriculture, food and related studies">Agriculture, food and related studies</option>' +
                '<option value="All subjects">All subjects</option>' +
                '<option value="Architecture, building and planning">Architecture, building and planning</option>' +
                '<option value="Biological and sport sciences">Biological and sport sciences</option>' +
                '<option value="Business and management">Business and management</option>' +
                '<option value="Combined and general studies">Combined and general studies</option>' +
                '<option value="Computing">Computing</option>' +
                '<option value="Design, and creative and performing arts">Design, and creative and performing arts</option>' +
                '<option value="Education and teaching">Education and teaching</option>' +
                '<option value="Engineering and technology">Engineering and technology</option>' +
                '<option value="Geography, earth and environmental studies">Geography, earth and environmental studies</option>' +
                '<option value="Historical, philosophical and religious studies">Historical, philosophical and religious studies</option>' +
                '<option value="Language and area studies">Language and area studies</option>' +
                '<option value="Law">Law</option>' +
                '<option value="Mathematical sciences">Mathematical sciences</option>' +
                '<option value="Media, journalism and communications">Media, journalism and communications</option>' +
                '<option value="Medicine and dentistry">Medicine and dentistry</option>' +
                '<option value="Physical sciences">Physical sciences</option>' +
                '<option value="Psychology">Psychology</option>' +
                '<option value="Social sciences">Social sciences</option>' +
                '<option value="Subjects allied to medicine">Subjects allied to medicine</option>' +
                '<option value="Veterinary sciences">Veterinary sciences</option>' +
                '</select>');

        } else if (selectedCharacteristic === 'Disability Status') {
            specificOptionsDropdown.append('<label for="dsOptions">Do you have a disability:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="dsOptions">' +
                '<option value="nothing">---</option>' +
                '<option value="Disability reported">Yes</option>' +
                '<option value="No disability reported">No</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="providerCountry">Provider Country:</label>');
            specificOptionsDropdown.append('<select class="form-control" id="providerCountry">' +
                '<option value="nothing">---</option>' +
                '<option value="England">England</option>' +
                '<option value="Northern Ireland">Northern Ireland</option>' +
                '<option value="Scotland">Scotland</option>' +
                '<option value="UK">UK</option>' +
                '<option value="Wales">Wales</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="MoS">Mode of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="MoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All modes">All modes</option>' +
                '<option value="Apprenticeship">Apprenticeship</option>' +
                '<option value="Full-time">Full-time</option>' +
                '<option value="Part-time">Part-time</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="LoS">Level of Study</label>');
            specificOptionsDropdown.append('<select class="form-control" id="LoS">' +
                '<option value="nothing">---</option>' +
                '<option value="All undergraduates">All undergraduates</option>' +
                '<option value="First degree">First degree</option>' +
                '<option value="Other undergraduate">Other undergraduate</option>' +
                '<option value="Undergraduate with postgraduate component">Undergraduate with postgraduate component</option>' +
                '</select>');
            specificOptionsDropdown.append('<label for="Subject">Subject</label>');
            specificOptionsDropdown.append('<select class="form-control" id="Subject">' +
                '<option value="nothing">---</option>' +
                '<option value="Agriculture, food and related studies">Agriculture, food and related studies</option>' +
                '<option value="All subjects">All subjects</option>' +
                '<option value="Architecture, building and planning">Architecture, building and planning</option>' +
                '<option value="Biological and sport sciences">Biological and sport sciences</option>' +
                '<option value="Business and management">Business and management</option>' +
                '<option value="Combined and general studies">Combined and general studies</option>' +
                '<option value="Computing">Computing</option>' +
                '<option value="Design, and creative and performing arts">Design, and creative and performing arts</option>' +
                '<option value="Education and teaching">Education and teaching</option>' +
                '<option value="Engineering and technology">Engineering and technology</option>' +
                '<option value="Geography, earth and environmental studies">Geography, earth and environmental studies</option>' +
                '<option value="Historical, philosophical and religious studies">Historical, philosophical and religious studies</option>' +
                '<option value="Language and area studies">Language and area studies</option>' +
                '<option value="Law">Law</option>' +
                '<option value="Mathematical sciences">Mathematical sciences</option>' +
                '<option value="Media, journalism and communications">Media, journalism and communications</option>' +
                '<option value="Medicine and dentistry">Medicine and dentistry</option>' +
                '<option value="Physical sciences">Physical sciences</option>' +
                '<option value="Psychology">Psychology</option>' +
                '<option value="Social sciences">Social sciences</option>' +
                '<option value="Subjects allied to medicine">Subjects allied to medicine</option>' +
                '<option value="Veterinary sciences">Veterinary sciences</option>' +
                '</select>');
        }

    });

    $('#predictionForm').submit(function(event) {
        event.preventDefault(); // Prevent default form submission
        
        // Extract selected characteristic and options
        var characteristic = $('#characteristic').val();
        var selectedOption;
        if (characteristic === 'Age') {
            split = $('#ageOptions').val();
            pc = $('#providerCountry').val();
            mos = $('#MoS').val();
            los = $('#LoS').val();
            sub = $('#Subject').val();
        } else if (characteristic === 'Sex') {
            split = $('#sexOptions').val();
            pc = $('#providerCountry').val();
            mos = $('#MoS').val();
            los = $('#LoS').val();
            sub = $('#Subject').val(); 
        } else if (characteristic === 'Ethnicity') {
            split = $('#ethnicityOptions').val(); 
            console.log("Split value:", split);
            pc = $('#providerCountry').val();
            mos = $('#MoS').val();
            los = $('#LoS').val();
            sub = $('#Subject').val();
        } else if (characteristic === 'Domicile') {
            split = $('#domicileOptions').val(); 
            pc = $('#providerCountry').val();
            mos = $('#MoS').val();
            los = $('#LoS').val();
            sub = $('#Subject').val();
        } else if (characteristic === 'Disability Status') {
            split = $('#dsOptions').val(); 
            pc = $('#providerCountry').val();
            mos = $('#MoS').val();
            los = $('#LoS').val();
            sub = $('#Subject').val();
        };
            
        // Create JSON object containing the data
        var data = {
            characteristic: characteristic,
            split: split,
            pc: pc,
            mos: mos, 
            los: los,
            sub: sub
        };
        
        // Send data to Flask backend
        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function(response) {
                // Clear any existing prediction result
                $('#predictionResult').empty();

                // Iterate over each prediction in the response
                response.forEach(function(prediction) {
                    // Construct HTML for displaying prediction
                    var predictionHTML = '<div class="prediction">' +
                                            '<p>' + prediction.theme + '</p>' +
                                            '<p>' + prediction.satisfaction + '</p>' +
                                         '</div>';
                    // Append prediction HTML to the result container
                    $('#predictionResult').append(predictionHTML);
                });
            },
            error: function(xhr, status, error) {
                // Handle error
                console.error('Error:', error); // Log error to console
            }
        });
    });
});

// REFERENCES
//[1]W3 Schools, “JavaScript Tutorial,” W3schools.com, 2019. https://www.w3schools.com/Js/
//[2]javaTpoint, “How to create dropdown list using JavaScript - javatpoint,” www.javatpoint.com. https://www.javatpoint.com/how-to-create-dropdown-list-using-javascript
//[3]olawanletjoel, “HTML Drop-down Menu – How to Add a Drop-Down List with the Select Element,” freeCodeCamp.org, Sep. 26, 2022. https://www.freecodecamp.org/news/html-drop-down-menu-how-to-add-a-drop-down-list-with-the-select-element/
//[4]Geek for Geek, “How to add options to a drop-down list using jQuery?,” GeeksforGeeks, Jan. 22, 2021. https://www.geeksforgeeks.org/how-to-add-options-to-a-drop-down-list-using-jquery/
//[5]Geeks for Geeks, “Pass JavaScript Variables to Python in Flask,” GeeksforGeeks, Apr. 17, 2023. https://www.geeksforgeeks.org/pass-javascript-variables-to-python-in-flask/ 