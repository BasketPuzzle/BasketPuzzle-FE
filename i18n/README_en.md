# BasketPuzzle Project

## Problem Identification and Analysis
In modern marketing, analyzing the products consumers put into their shopping carts is a critical task. Understanding consumer purchasing patterns and identifying frequently bought-together products are essential.
The goal of this project is to analyze consumer basket patterns, identify product combinations that lead to successful transactions, and establish effective marketing strategies.
To differentiate this project from similar existing studies, we aim to implement not only customized product package recommendations for different groups but also an algorithm that outputs sales volumes and related product information based on specific product name searches. This approach will provide greater value and insights for practical business strategies.

## Project Objectives
The objective of BasketPuzzle is to analyze consumer purchase histories to discover patterns among products and represent them as quantifiable data.
We plan to build an API pipeline and visualize the analysis results on the web to make them easily accessible to users. This will enable sellers to develop more effective marketing strategies and improve customer satisfaction, while consumers can enjoy a better shopping experience through personalized recommendations.

## Key Features
1. Visualization of shopping cart data in charts
2. Provision of shopping trend lists in charts
3. Product information output based on specific product name searches (e.g., sales volume, top 3 related products)
4. Analysis and recommendation of related products based on user input

## Development Tools and Languages
Data Analysis: Conducted in a Python-based Jupyter Notebook environment using libraries like Pandas, NumPy, and Matplotlib for data analysis, preprocessing, and computation.
Association Rule Learning: Utilized the Apriori algorithm for frequent pattern mining.
Backend API: Built with Flask to serve analysis data as APIs.
Web Visualization: Implemented with HTML, CSS, JavaScript to visualize the analysis results on the web.
