package ch.ethz.idsc.subare.ch04.gambler;

import java.util.Deque;
import java.util.LinkedList;

import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.red.Min;
import ch.ethz.idsc.tensor.sca.N;
import ch.ethz.idsc.tensor.sca.Power;

public class LearningRateScheduler {
  double exponent;
  double factor;
  double alpha; // RealScalar.ONE;
  double epsilon;
  Deque<Scalar> errors = new LinkedList<>();

  public LearningRateScheduler(double exponent, double factor, double alpha, double epsilon) {
    this.exponent = exponent;
    this.factor = factor;
    this.alpha = alpha; 
    this.epsilon = epsilon;
  }

  Scalar getRate(int index) {
    return Power.of(N.of(DoubleScalar.of(1 / (factor * (index + 1)))), this.exponent);
  }
  
  Scalar getRate() {
    return RealScalar.of(alpha);
  }

  void notifyError(Scalar error) {
    errors.add(error);
    if (errors.size() == 2) {
      Scalar error_prev = errors.poll(); // n-5
      Scalar error_min = errors.stream().reduce(Min::of).get();
      if (Scalars.lessThan(error_prev, error_min)) {
        alpha /= 2;
        epsilon /= 2;
        System.out.println("Current alpha: " + alpha);
        errors.clear();
      }
    }
  }

  public Scalar getEpsilon() {
    // TODO Auto-generated method stub
    return RealScalar.of(epsilon);
  }
}
