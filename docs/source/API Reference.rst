API Reference
=================
- **`POST /api/upload`**
  - **Description:** Uploads purchase history data.
  - **Request Body:**
    .. code-block:: json

       {
         "userId": "User ID",
         "data": "CSV Data"
       }

  - **Response:** Upload status message.

- **`GET /api/analysis`**
  - **Description:** Fetches analysis results.
  - **Response:** Analysis result data.
