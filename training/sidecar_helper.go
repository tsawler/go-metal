package training

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// SidecarManager handles automatic sidecar service management
type SidecarManager struct {
	sidecarPath   string
	port          int
	autoStart     bool
	dockerMode    bool
	process       *exec.Cmd
	isRunning     bool
}

// SidecarConfig contains configuration for the sidecar service
type SidecarConfig struct {
	Port       int    `json:"port"`
	AutoStart  bool   `json:"auto_start"`
	DockerMode bool   `json:"docker_mode"`
	SidecarDir string `json:"sidecar_dir"`
}

// DefaultSidecarConfig returns default configuration for the sidecar
func DefaultSidecarConfig() SidecarConfig {
	return SidecarConfig{
		Port:       8080,
		AutoStart:  true,
		DockerMode: false,
		SidecarDir: "", // Will be auto-detected
	}
}

// NewSidecarManager creates a new sidecar manager
func NewSidecarManager(config SidecarConfig) (*SidecarManager, error) {
	// Auto-detect sidecar directory if not provided
	sidecarPath := config.SidecarDir
	if sidecarPath == "" {
		detected, err := detectSidecarPath()
		if err != nil {
			return nil, fmt.Errorf("failed to detect sidecar path: %w", err)
		}
		sidecarPath = detected
	}
	
	// Verify sidecar exists
	if !fileExists(filepath.Join(sidecarPath, "app.py")) {
		return nil, fmt.Errorf("sidecar app.py not found at %s", sidecarPath)
	}
	
	return &SidecarManager{
		sidecarPath: sidecarPath,
		port:        config.Port,
		autoStart:   config.AutoStart,
		dockerMode:  config.DockerMode,
		isRunning:   false,
	}, nil
}

// Start starts the sidecar service
func (sm *SidecarManager) Start() error {
	if sm.isRunning {
		return nil // Already running
	}
	
	// Check if service is already running elsewhere
	if sm.isServiceRunning() {
		sm.isRunning = true
		return nil
	}
	
	if !sm.autoStart {
		return fmt.Errorf("sidecar service not running and auto-start is disabled")
	}
	
	// Start the service
	if sm.dockerMode {
		return sm.startDockerService()
	}
	return sm.startPythonService()
}

// Stop stops the sidecar service
func (sm *SidecarManager) Stop() error {
	if !sm.isRunning {
		return nil
	}
	
	if sm.dockerMode {
		return sm.stopDockerService()
	}
	return sm.stopPythonService()
}

// IsRunning checks if the sidecar service is running
func (sm *SidecarManager) IsRunning() bool {
	return sm.isServiceRunning()
}

// GetBaseURL returns the base URL for the sidecar service
func (sm *SidecarManager) GetBaseURL() string {
	return fmt.Sprintf("http://localhost:%d", sm.port)
}

// EnsureRunning ensures the sidecar service is running
func (sm *SidecarManager) EnsureRunning() error {
	if sm.isServiceRunning() {
		return nil
	}
	
	fmt.Printf("🚀 Starting Go-Metal visualization sidecar...\n")
	
	if err := sm.Start(); err != nil {
		return fmt.Errorf("failed to start sidecar service: %w", err)
	}
	
	// Wait for service to be ready
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for sidecar service to start")
		default:
			if sm.isServiceRunning() {
				fmt.Printf("✅ Sidecar service is running at %s\n", sm.GetBaseURL())
				return nil
			}
			time.Sleep(500 * time.Millisecond)
		}
	}
}

// startPythonService starts the Python development server
func (sm *SidecarManager) startPythonService() error {
	// Check if Python is available
	pythonCmd := "python3"
	if runtime.GOOS == "windows" {
		pythonCmd = "python"
	}
	
	// Check if start_sidecar.py exists
	startScript := filepath.Join(sm.sidecarPath, "start_sidecar.py")
	if fileExists(startScript) {
		// Use the startup script
		cmd := exec.Command(pythonCmd, "start_sidecar.py", "--dev", "--port", fmt.Sprintf("%d", sm.port))
		cmd.Dir = sm.sidecarPath
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		
		sm.process = cmd
		sm.isRunning = true
		
		return cmd.Start()
	}
	
	// Fallback to direct app.py execution
	cmd := exec.Command(pythonCmd, "app.py")
	cmd.Dir = sm.sidecarPath
	cmd.Env = append(os.Environ(), fmt.Sprintf("PORT=%d", sm.port))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	
	sm.process = cmd
	sm.isRunning = true
	
	return cmd.Start()
}

// startDockerService starts the Docker Compose service
func (sm *SidecarManager) startDockerService() error {
	// Check if Docker Compose is available
	if !commandExists("docker-compose") {
		return fmt.Errorf("docker-compose command not found")
	}
	
	// Start with Docker Compose
	cmd := exec.Command("docker-compose", "up", "-d")
	cmd.Dir = sm.sidecarPath
	
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to start Docker service: %w\nOutput: %s", err, string(output))
	}
	
	sm.isRunning = true
	return nil
}

// stopPythonService stops the Python development server
func (sm *SidecarManager) stopPythonService() error {
	if sm.process != nil {
		if err := sm.process.Process.Kill(); err != nil {
			return fmt.Errorf("failed to stop Python service: %w", err)
		}
		sm.process = nil
	}
	
	sm.isRunning = false
	return nil
}

// stopDockerService stops the Docker Compose service
func (sm *SidecarManager) stopDockerService() error {
	cmd := exec.Command("docker-compose", "down")
	cmd.Dir = sm.sidecarPath
	
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to stop Docker service: %w\nOutput: %s", err, string(output))
	}
	
	sm.isRunning = false
	return nil
}

// isServiceRunning checks if the sidecar service is responding
func (sm *SidecarManager) isServiceRunning() bool {
	client := &http.Client{
		Timeout: 2 * time.Second,
	}
	
	resp, err := client.Get(fmt.Sprintf("%s/health", sm.GetBaseURL()))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	
	return resp.StatusCode == http.StatusOK
}

// detectSidecarPath attempts to find the sidecar directory
func detectSidecarPath() (string, error) {
	// Common locations to check
	candidates := []string{
		"./app/sidecar",
		"../sidecar",
		"../../app/sidecar",
		"./sidecar",
	}
	
	// Get current working directory
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	
	// Check candidates relative to current directory
	for _, candidate := range candidates {
		fullPath := filepath.Join(cwd, candidate)
		if fileExists(filepath.Join(fullPath, "app.py")) {
			abs, err := filepath.Abs(fullPath)
			if err == nil {
				return abs, nil
			}
		}
	}
	
	// Try to find in Go module structure
	// Look for go.mod and then find sidecar relative to it
	dir := cwd
	for {
		goMod := filepath.Join(dir, "go.mod")
		if fileExists(goMod) {
			// Found go.mod, check for sidecar
			sidecarPath := filepath.Join(dir, "app", "sidecar")
			if fileExists(filepath.Join(sidecarPath, "app.py")) {
				return sidecarPath, nil
			}
			break
		}
		
		parent := filepath.Dir(dir)
		if parent == dir {
			break // Reached root
		}
		dir = parent
	}
	
	return "", fmt.Errorf("sidecar directory not found in common locations")
}

// fileExists checks if a file exists
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// commandExists checks if a command exists in PATH
func commandExists(cmd string) bool {
	_, err := exec.LookPath(cmd)
	return err == nil
}

// Helper methods for ModelTrainer integration

// EnableSidecarWithAutoStart enables plotting service with automatic sidecar management
func (mt *ModelTrainer) EnableSidecarWithAutoStart(config ...SidecarConfig) error {
	// Use default config if none provided
	sidecarConfig := DefaultSidecarConfig()
	if len(config) > 0 {
		sidecarConfig = config[0]
	}
	
	// Create sidecar manager
	manager, err := NewSidecarManager(sidecarConfig)
	if err != nil {
		return fmt.Errorf("failed to create sidecar manager: %w", err)
	}
	
	// Ensure sidecar is running
	if err := manager.EnsureRunning(); err != nil {
		return fmt.Errorf("failed to start sidecar: %w", err)
	}
	
	// Configure plotting service to use the sidecar
	mt.ConfigurePlottingService(PlottingServiceConfig{
		BaseURL: manager.GetBaseURL(),
		Timeout: 30 * time.Second,
	})
	
	// Enable services
	mt.EnableVisualization()
	mt.EnablePlottingService()
	
	return nil
}

// GenerateAndOpenPlot generates a plot and opens it in the browser
func (mt *ModelTrainer) GenerateAndOpenPlot(plotType PlotType) error {
	// Generate plot
	response, err := mt.SendPlotToSidecar(plotType)
	if err != nil {
		return fmt.Errorf("failed to generate plot: %w", err)
	}
	
	if !response.Success {
		return fmt.Errorf("plot generation failed: %s", response.Message)
	}
	
	// Open in browser
	baseURL := strings.TrimSuffix(mt.plottingService.baseURL, "/")
	openURL := fmt.Sprintf("%s/api/open/%s", baseURL, response.PlotURL[6:]) // Remove "/plot/" prefix
	
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(openURL)
	if err != nil {
		return fmt.Errorf("failed to open plot in browser: %w", err)
	}
	defer resp.Body.Close()
	
	fmt.Printf("🌐 Plot opened in browser: %s\n", response.PlotURL)
	return nil
}

// GenerateAndOpenAllPlots generates all plots and opens them in a dashboard
func (mt *ModelTrainer) GenerateAndOpenAllPlots() error {
	// Generate all plots
	responses := mt.SendAllPlotsToSidecar()
	
	// Count successful plots
	successCount := 0
	var firstBatchID string
	
	for plotType, response := range responses {
		if response.Success {
			successCount++
			if firstBatchID == "" {
				// Extract batch ID from the first successful response
				// This is a simplified approach - in practice you'd want to implement proper batch handling
				firstBatchID = fmt.Sprintf("batch_%d", time.Now().Unix())
			}
			fmt.Printf("✅ Generated %s plot\n", plotType)
		} else {
			fmt.Printf("❌ Failed to generate %s plot: %s\n", plotType, response.Message)
		}
	}
	
	if successCount == 0 {
		return fmt.Errorf("no plots were generated successfully")
	}
	
	fmt.Printf("🎉 Generated %d plots successfully\n", successCount)
	return nil
}