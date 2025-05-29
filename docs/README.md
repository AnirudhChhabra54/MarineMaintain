# 🚢 SeaLogix: AI-Driven Ship Maintenance System

SeaLogix is an advanced maritime maintenance platform that leverages AI to detect anomalies in ship logs and provide predictive maintenance insights. The system processes various types of logs (CSV, PDF, images) and uses machine learning to identify potential issues before they become critical.

## 🌟 Features

- **Multi-format Log Ingestion**
  - CSV file processing
  - PDF text extraction
  - OCR for handwritten logs
  - Real-time log monitoring

- **AI-Powered Analysis**
  - Anomaly detection using Autoencoder and Isolation Forest
  - Predictive maintenance recommendations
  - Real-time alerting system
  - Feature importance analysis

- **Interactive Dashboard**
  - Real-time monitoring
  - Historical data visualization
  - Alert management
  - Maintenance scheduling

## 🛠️ Technology Stack

- **Frontend**
  - Next.js
  - React
  - Tailwind CSS
  - ShadCN UI Components

- **Backend**
  - FastAPI
  - PyTorch (Autoencoder)
  - Scikit-learn (Isolation Forest)
  - Tesseract OCR

- **Data Processing**
  - Pandas
  - NumPy
  - PyTesseract
  - PyPDF2

- **Deployment**
  - Docker
  - Docker Compose
  - GitHub Actions

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 20+
- Docker and Docker Compose
- Tesseract OCR

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sealogix.git
   cd sealogix
   ```

2. **Set up the backend**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   npm install
   ```

4. **Start the development servers**
   ```bash
   # Start the backend
   uvicorn backend.main:app --reload --port 8000
   
   # Start the frontend (in a new terminal)
   npm run dev
   ```

### Using Docker

1. **Build and run with Docker Compose**
   ```bash
   docker-compose -f deployment/docker-compose.yml up --build
   ```

2. **Access the services**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📊 System Architecture

```plaintext
                   +--------------------+
                   |   Log Upload (UI)  |
                   +---------+----------+
                             |
                             v
                +------------+-------------+
                |     FastAPI Backend      |
                +------------+-------------+
                             |
           +----------------+---------------------+
           |                |                     |
           v                v                     v
+----------------+  +-----------------+   +-----------------+
| ingest_csv.py  |  | ingest_pdf.py   |   | ingest_images.py|
+----------------+  +-----------------+   +-----------------+
           |                |                     |
           v                v                     v
        Preprocessed structured & unstructured data (text)
                             |
                             v
              +-----------------------------+
              |     ML Pipeline (Training)  |
              |  - Autoencoder              |
              |  - Isolation Forest         |
              +-------------+---------------+
                            |
                            v
                 +----------------------+
                 |   Inference Engine   |
                 | (anomaly scores etc) |
                 +----------+-----------+
                            |
                            v
               +----------------------------+
               |  React Dashboard (UI)      |
               |  - Maps / Charts / Tables  |
               +----------------------------+
```

## 🧪 Testing

Run the test suite:

```bash
# Backend tests
pytest tests/

# Frontend tests
npm run test
```

## 📦 Deployment

1. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Update the variables with your production values

2. **Deploy using Docker**
   ```bash
   docker-compose -f deployment/docker-compose.yml -f deployment/docker-compose.prod.yml up -d
   ```

3. **Monitor the services**
   ```bash
   docker-compose ps
   docker-compose logs -f
   ```

## 🔒 Security Considerations

- Use proper authentication in production
- Configure CORS settings
- Secure sensitive environment variables
- Regular security updates
- Input validation and sanitization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
- etc
