// code by jph
package ch.ethz.idsc.subare.ch02;

import java.util.Random;

import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;

public abstract class Agent {
  protected static final Random random = new Random();
  // ---
  private Scalar total = ZeroScalar.get();
  private Integer count = 0;
  private int count_copy;
  private Tensor actions = Tensors.empty();
  private Tensor qvalues = Tensors.empty();

  public abstract int takeAction();

  // shall only be used to recording history
  protected abstract Tensor protected_QValues();

  // shall not call getCount() from within
  protected abstract void protected_feedback(int a, Scalar value);

  public final void feedback(int a, Scalar value) {
    total = total.add(value);
    actions.append(RealScalar.of(a));
    ++count;
    count_copy = count;
    count = null; // prevent functions to use getCount()
    qvalues.append(protected_QValues());
    protected_feedback(a, value);
    count = count_copy;
  }

  public final Scalar getCount() {
    return RealScalar.of(count);
  }

  public final Scalar getRewardTotal() {
    return total;
  }

  public final Scalar getRewardAverage() {
    return total.divide(RealScalar.of(count_copy));
  }

  public final Tensor getActions() {
    return actions.unmodifiable();
  }

  public final Tensor getQValues() {
    return qvalues.unmodifiable();
  }

  public abstract String getDescription();

  public final String getAbsDesc() {
    String name = getClass().getSimpleName();
    name = name.substring(0, name.length() - 5); // drop "Agent"
    return String.format("%10s%15s", name, getDescription());
  }

  @Override
  public String toString() {
    return getAbsDesc();
  }
}
