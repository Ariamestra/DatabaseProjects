CREATE DATABASE occupations;

\c occupations

DROP TABLE IF EXISTS Occupations;

-- -------------------------------------------------------------------------------------------------------

-- Create table Occupations
CREATE TABLE Occupations (
    Code VARCHAR(10),
    Occupation VARCHAR(150),
    JobFamily VARCHAR(50)
);

-- Populate table Occupations
\copy Occupations from '/tmp/occupations.csv' CSV HEADER;

-- Total number of occupations
SELECT COUNT(*) 
FROM Occupations;


-- List of all job families in alphabetical order
SELECT DISTINCT JobFamily 
FROM Occupations 
ORDER BY JobFamily;


-- Total number of job families 
SELECT JobFamily, COUNT(*) AS total 
From Occupations 
GROUP By JobFamily;

-- Total number of occupations per job family in alphabetical order of job family.
SELECT JobFamily, COUNT(*) AS NumberOfOccupations 
FROM Occupations 
GROUP BY JobFamily 
ORDER BY JobFamily;


-- Number of occupations in the "Computer and Mathematical" job family 
SELECT JobFamily, COUNT(*) 
FROM Occupations 
WHERE JobFamily = 'Computer and Mathematical' 
GROUP BY JobFamily;


-- Alphabetical list of occupations in the "Computer and Mathematical" job family.
SELECT Occupation 
FROM Occupations 
WHERE JobFamily = 'Computer and Mathematical' 
ORDER BY Occupation;

-- Alphabetical list of occupations in the "Computer and Mathematical" job family that begins with the word "Database"
SELECT Occupation 
FROM Occupations 
WHERE JobFamily = 'Computer and Mathematical' AND Occupation LIKE 'Database%' 
ORDER BY Occupation;