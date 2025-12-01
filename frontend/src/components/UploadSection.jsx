import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileUp, CheckCircle, AlertCircle, Camera } from 'lucide-react';
import axios from 'axios';
import CameraCapture from './CameraCapture';

const UploadSection = ({ setResults, setLoading }) => {
    const [files, setFiles] = useState([]);
    const [answerKey, setAnswerKey] = useState(null);
    const [showCamera, setShowCamera] = useState(false);
    const [uploadStatus, setUploadStatus] = useState('');

    const onDrop = useCallback(acceptedFiles => {
        setFiles(prev => [...prev, ...acceptedFiles]);
    }, []);

    const onKeyDrop = useCallback(acceptedFiles => {
        if (acceptedFiles.length > 0) {
            setAnswerKey(acceptedFiles[0]);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'image/*': [], 'application/pdf': [] } });
    const { getRootProps: getKeyRoot, getInputProps: getKeyInput } = useDropzone({
        onDrop: onKeyDrop,
        accept: { 'text/csv': ['.csv'] },
        maxFiles: 1
    });

    const handleProcess = async () => {
        if (!answerKey || files.length === 0) {
            alert("Please upload both Answer Key and OMR Sheets.");
            return;
        }

        setLoading(true);
        setResults(null);
        setUploadStatus('Uploading files...');

        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });

        try {
            // 1. Upload Files
            await axios.post('http://localhost:8000/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            setUploadStatus('Processing...');

            // 2. Process
            const processData = new FormData();
            processData.append('answer_key', answerKey);

            const response = await axios.post('http://localhost:8000/process', processData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            console.log("Backend Response:", response.data);

            if (response.data.status === 'failed' || response.data.status === 'error') {
                setUploadStatus(`Error: ${response.data.message}`);
                alert(`Processing failed: ${response.data.message}`);
                return;
            }

            if (!response.data.data || response.data.data.length === 0) {
                setUploadStatus('No results found.');
                alert("No results were generated. Please check your inputs.");
                return;
            }

            setResults(response.data.data);
            setUploadStatus('Done!');
        } catch (error) {
            console.error("Error:", error);
            setUploadStatus('Error occurred.');
            alert("Processing failed. Check console.");
        } finally {
            setLoading(false);
        }
    };

    const handleCameraCapture = (file) => {
        setFiles(prev => [...prev, file]);
        setShowCamera(false);
    };

    return (
        <div className="space-y-8">
            <div className="flex flex-col md:flex-row gap-6">
                {/* Answer Key Upload */}
                <div className="flex-1">
                    <label className="block text-sm font-medium text-gray-700 mb-2">1. Upload Answer Key (CSV)</label>
                    <div {...getKeyRoot()} className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${answerKey ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-blue-400'}`}>
                        <input {...getKeyInput()} />
                        {answerKey ? (
                            <div className="flex items-center justify-center text-green-700 gap-2">
                                <CheckCircle size={20} />
                                <span className="font-medium">{answerKey.name}</span>
                            </div>
                        ) : (
                            <div className="text-gray-400">
                                <FileUp className="mx-auto mb-2" size={32} />
                                <p>Drag & drop CSV here</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* OMR Sheets Upload */}
                <div className="flex-[2]">
                    <label className="block text-sm font-medium text-gray-700 mb-2">2. Upload OMR Sheets (Images/PDF)</label>
                    <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center transition-colors hover:border-blue-400 relative">
                        <div {...getRootProps()} className="cursor-pointer">
                            <input {...getInputProps()} />
                            <Upload className="mx-auto mb-2 text-gray-400" size={32} />
                            <p className="text-gray-500 mb-2">Drag & drop files here, or click to select</p>
                            <p className="text-xs text-gray-400">{files.length} files selected</p>
                        </div>

                        <div className="absolute top-4 right-4">
                            <button
                                onClick={(e) => { e.stopPropagation(); setShowCamera(true); }}
                                className="bg-blue-100 hover:bg-blue-200 text-blue-700 p-2 rounded-full transition-colors"
                                title="Use Camera"
                            >
                                <Camera size={20} />
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {showCamera && (
                <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-2xl p-4 max-w-2xl w-full">
                        <CameraCapture onCapture={handleCameraCapture} onClose={() => setShowCamera(false)} />
                    </div>
                </div>
            )}

            <div className="text-center">
                <button
                    onClick={handleProcess}
                    disabled={!answerKey || files.length === 0}
                    className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-full shadow-lg transform transition hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    Start Processing
                </button>
                {uploadStatus && <p className="mt-2 text-sm text-gray-500">{uploadStatus}</p>}
            </div>
        </div>
    );
};

export default UploadSection;
