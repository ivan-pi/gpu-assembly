__global__ void bgk_kernel_split(const int n, const float omega, float *pdf, float *rho, float *ux, float *uy, float *indp) {
   int i = blockDim.x * blockIdx.x + threadIdx.x;

   if (i < n) {
      float w[3] = {4. / 9., 1. / 9., 1. / 36.};
      float omegabar = 1.0 - omega;
      float one_third = 1.0 / 3.0;
      float one_half = 1.0 / 2.0;
      float th = 3.0 / 2.0;

      float omega_w0 = 3.0 * omega * w[0];
      float omega_ws = 3.0 * omega * w[1];
      float omega_wd = 3.0 * omega * w[2];

      float f[9];

      for (int a = 0; a < 9; ++a) {
         int ia = i + n * a;
         f[a] = pdf[ia];
      }

      float rho_temp = 0.;
      for (int a = 0; a < 9; ++a) {
         rho_temp += f[a];
      }

      rho[i] = rho_temp;

      float invrho = 1.0 / rho_temp;

      float ux_temp = invrho * (((f[5] - f[7]) + (f[8] - f[6])) + (f[1] - f[3]));
      float uy_temp = invrho * (((f[5] - f[7]) + (f[6] - f[8])) + (f[2] - f[4]));

      ux[i] = ux_temp;
      uy[i] = uy_temp;

      float uxsq = ux_temp * ux_temp;
      float uysq = uy_temp * uy_temp;

      float indp_temp = one_third - one_half * (uxsq + uysq);
      indp[i] = indp_temp;

      int ia = i;
      pdf[ia] = omegabar * pdf[ia] + omega_w0 * rho_temp * indp_temp;

      float vel_trm_13 = indp_temp + th * uxsq;

      ia = i + n;
      pdf[ia] = omegabar * pdf[ia] + omega_ws * rho_temp * (vel_trm_13 + ux_temp);

      ia = i + n * 3;
      pdf[ia] = omegabar * pdf[ia] + omega_ws * rho_temp * (vel_trm_13 - ux_temp);

      float vel_trm_24 = indp_temp + th * uysq;

      ia = i + n * 2;
      pdf[ia] = omegabar * pdf[ia] + omega_ws * rho_temp * (vel_trm_24 + uy_temp);

      ia = i + n * 4;
      pdf[ia] = omegabar * pdf[ia] + omega_ws * rho_temp * (vel_trm_24 - uy_temp);

      float velxpy = ux_temp + uy_temp;
      float vel_trm_57 = indp_temp + th * velxpy * velxpy;

      ia = i + n * 5;
      pdf[ia] = omegabar * pdf[ia] + omega_wd * rho_temp * (vel_trm_57 + velxpy);

      ia = i + n * 7;
      pdf[ia] = omegabar * pdf[ia] + omega_wd * rho_temp * (vel_trm_57 - velxpy);

      float velxmy = ux_temp - uy_temp;
      float vel_trm_68 = indp_temp + th * velxmy * velxmy;

      ia = i + n * 6;
      pdf[ia] = omegabar * pdf[ia] + omega_wd * rho_temp * (vel_trm_68 - velxmy);

      ia = i + n * 8;
      pdf[ia] = omegabar * pdf[ia] + omega_wd * rho_temp * (vel_trm_68 + velxmy);
   }
}