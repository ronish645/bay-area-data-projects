MySQL-based Driving School Management System project demonstrating normalization, relationships, and analytical SQL queries.

# Driving School Database & SQL Analytics Project

This project demonstrates the design, normalization, and implementation of a **Driving School Management System** using MySQL.
It models real-world entities such as students, instructors, lessons, vehicles, and payments — and applies advanced SQL queries to extract actionable insights.

---

## Objectives

* Build a **relational database** for a driving school.
* Normalize data up to **BCNF** to eliminate redundancy.
* Populate tables with realistic sample data.
* Execute **analytical SQL queries** for business insights.
* Document the design process through ER modeling and relational mapping.

---

##  Database Schema

**Key Entities:**

| Table         | Description                                       |
| ------------- | ------------------------------------------------- |
| `Office`      | Stores driving school branches and departments.   |
| `Employee`    | Holds details of staff and instructors.           |
| `Student`     | Manages student records and registration data.    |
| `Instructor`  | Represents certified trainers linked to vehicles. |
| `Vehicle`     | Vehicle inventory and maintenance data.           |
| `Course`      | Scheduling details for lessons and sessions.      |
| `Maintenance` | Tracks servicing costs and dates.                 |
| `Payment`     | Records student payments and fee structures.      |

Each table includes **primary keys**, **foreign keys**, and **constraints** for referential integrity.

---

## Key Features

* Database normalized to **BCNF (Boyce-Codd Normal Form)**
* Inserted realistic sample data for demonstration
* Used **joins, aggregate functions, subqueries, and views**
* Created **indexes** for faster lookup on foreign keys
* Applied **CHECK** and **DEFAULT** constraints for data consistency

---

## 💡 Example Queries

```sql
-- Total revenue per instructor
SELECT i.FirstName, i.LastName, SUM(p.Amount) AS TotalRevenue
FROM Instructor i
JOIN Course c ON i.InstructorID = c.InstructorID
JOIN Payment p ON c.CourseID = p.CourseID
GROUP BY i.InstructorID;

-- Average maintenance cost by vehicle type
SELECT v.VehicleType, ROUND(AVG(m.Cost), 2) AS AvgMaintenanceCost
FROM Vehicle v
JOIN Maintenance m ON v.VehicleID = m.VehicleID
GROUP BY v.VehicleType;

-- Students older than 25 years
SELECT FirstName, LastName, YEAR(CURDATE()) - YEAR(DOB) AS Age
FROM Student
WHERE YEAR(CURDATE()) - YEAR(DOB) > 25;
```

---

## Insights & Results

* Full-time instructors using **Sedans** completed the most lessons.
* **SUVs** had the highest average maintenance cost.
* Top 20% of instructors contributed to over 60% of total revenue.
* Department with most employees also had the highest student enrollment.

---

## Tools & Technologies

| Category    | Tools                                 |
| ----------- | ------------------------------------- |
| Database    | MySQL / MariaDB                       |
| Language    | SQL (DDL, DML, DQL)                   |
| Environment | MySQL Workbench / XAMPP               |
| Design      | ER Diagram → Relational Schema → BCNF |
| Docs        | Project Reports (Modules 4 & 5 PDFs)  |

---

## Project Structure

```
driving_school_analysis/
├── Proj_Module4.sql     # Database creation & data insertion
├── Proj_Module5.sql     # Normalization & analytical queries
├── Proj_Module4.pdf     # ER model and relational mapping
├── Proj_Module5.pdf     # BCNF normalization & data dictionary
└── README.md            # Project documentation (this file)
```

---

## Future Enhancements

* Add **stored procedures** for automated data validation.
* Create **views** for instructor performance dashboards.
* Connect to **Python (Pandas/Matplotlib)** for visualization.
* Integrate **triggers** for maintenance and payment alerts.

---

## Author

Ronish Shrestha
📍 Fremont, California
💻 [GitHub](https://github.com/ronish645) | 🔗 [LinkedIn](https://linkedin.com/in/ronishshrestha)
📧 [ronishshrestha04@email.com](mailto:ronishshrestha04@email.com)

If you found this project helpful, please give the repo a star!*
