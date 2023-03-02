// 四元数を用いた剛体の回転運動シミュレーション

use nalgebra::geometry::{Quaternion, UnitQuaternion};
use nalgebra::{SMatrix, SVector};
use std::fs;
use std::io::{BufWriter, Write};

const DT: f64 = 0.03125; // 32Hz
                         // const DT: f64 = 0.0166666666666666667;
                         // const DT: f64 = (1/60) as f64;
const TIME_RANGE: f64 = 30.0; // [s]
const SPLIT_NUM: usize = (TIME_RANGE / DT) as usize + 1;
const OUT_FILE_PATH: &'static str = "data/result.csv";

type SVector3 = SVector<f64, 3>;
type SMatrix3x3 = SMatrix<f64, 3, 3>;

fn main() {
    // シミュレーション結果の保存先（同一ファイルが存在したら上書き）
    let mut file = BufWriter::new(fs::File::create(OUT_FILE_PATH).unwrap());

    // トルク
    let torque = Quaternion::new(0.0, 0.0, 0.0, 0.0); // 外力無し
                                                      // 初期角速度[rad/s]
    let gyr_init = SVector3::new(0.01, 8.0, 0.01);
    // 慣性テンソル（慣性モーメントの単位は[kg m^2]）
    let inertia_tensor = 1e-6 * SMatrix3x3::from_diagonal(&SVector3::new(62.2, 171.5, 210.5));
    // 慣性座標系上における角運動量
    let inertial_angular_momentum_vec = inertia_tensor * gyr_init;
    let mut inertial_angular_momentum = Quaternion::from_imag(inertial_angular_momentum_vec);
    // 四元数
    let axis = UnitQuaternion::identity();
    let mut q = *axis.quaternion();
    assert_eq!(q, Quaternion::new(1.0, 0.0, 0.0, 0.0));

    let mut pre_gyr = gyr_init;

    for i in 0..SPLIT_NUM {
        // 表示用
        // let body_angular_momentum: SVector3 = quat::frame_rotation(q, inertial_angular_momentum.into()).into(); // 座標系回転
        let body_angular_momentum: SVector3 = (q.conjugate() * inertial_angular_momentum * q)
            .imag()
            .into();
        let body_gyr = inertia_tensor.try_inverse().unwrap() * body_angular_momentum;
        let alpha = (body_gyr - pre_gyr) / DT; // 角加速度
        pre_gyr = body_gyr;

        // ダイナミクス
        inertial_angular_momentum = inertial_angular_momentum + DT * torque;

        // キネマティクス
        q = rk4(inertia_tensor, inertial_angular_momentum, q);
        q = q.normalize();

        // シミュレーション結果保存
        let unitq = UnitQuaternion::from_quaternion(q);
        let rpy = unitq.euler_angles();

        file.write(
            format!(
                "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.7},{:.7},{:.7}\n",
                DT * i as f64,
                rpy.2,
                rpy.1,
                rpy.0,
                body_gyr[0],
                body_gyr[1],
                body_gyr[2],
                alpha[0],
                alpha[1],
                alpha[2],
                body_angular_momentum[0],
                body_angular_momentum[1],
                body_angular_momentum[2]
            )
            .as_bytes(),
        )
        .unwrap();
    }

    println!("Result has been saved to {}", OUT_FILE_PATH);
}

/// 4次のルンゲ・クッタ法
fn rk4(
    inertia_tensor: SMatrix3x3,
    inertial_angular_momentum: Quaternion<f64>,
    q: Quaternion<f64>,
) -> Quaternion<f64> {
    let f1 = dq_dt(inertia_tensor, inertial_angular_momentum, q);
    let f2 = dq_dt(
        inertia_tensor,
        inertial_angular_momentum,
        Quaternion::from_real(DT * 0.5) * f1 + q,
    );
    let f3 = dq_dt(
        inertia_tensor,
        inertial_angular_momentum,
        Quaternion::from_real(DT * 0.5) * f2 + q,
    );
    let f4 = dq_dt(
        inertia_tensor,
        inertial_angular_momentum,
        Quaternion::from_real(DT) * f3 + q,
    );
    let tmp1 = Quaternion::from_real(2.0) * f2 + f1;
    let tmp2 = Quaternion::from_real(2.0) * f3 + f4;
    Quaternion::from_real(DT / 6.0) * (tmp1 + tmp2) + q
}

/// 四元数の時間微分
fn dq_dt(
    inertia_tensor: SMatrix3x3,
    inertial_angular_momentum: Quaternion<f64>,
    q: Quaternion<f64>,
) -> Quaternion<f64> {
    let body_angular_momentum: SVector3 = (q.conjugate() * inertial_angular_momentum * q)
        .imag()
        .into();

    let body_gyr = inertia_tensor.try_inverse().unwrap() * body_angular_momentum;
    // quat::scale(0.5, quat::mul(q, (0.0, body_gyr.into())))
    Quaternion::from_real(0.5) * q * Quaternion::from_imag(body_gyr.into())
}

// #[warn(dead_code)]
// fn type_of<T>(_: T) -> String {
//     let a = std::any::type_name::<T>();
//     return a.to_string();
// }
