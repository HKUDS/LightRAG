import { NavigateFunction } from 'react-router-dom';

class NavigationService {
  private navigate: NavigateFunction | null = null;

  setNavigate(navigate: NavigateFunction) {
    this.navigate = navigate;
  }

  navigateToLogin() {
    if (this.navigate) {
      this.navigate('/login');
    }
  }
}

export const navigationService = new NavigationService();
