// code by jz
package ch.ethz.idsc.subare.ch04.gambler;

import java.util.Deque;
import java.util.LinkedList;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.red.Min;

class LearningRateDeque {
  private double alpha;
  private double epsilon;
  private final Deque<Scalar> errors = new LinkedList<>();

  public LearningRateDeque(double alpha, double epsilon) {
    this.alpha = alpha;
    this.epsilon = epsilon;
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
    return RealScalar.of(epsilon);
  }
}
