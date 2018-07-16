// code by fluric
package ch.ethz.idsc.subare.util;

import java.util.Random;

import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.sca.Clip;

public class Coinflip {
  /** @param p_head
   * @return instance of Coinflip with given probability p_head that tossHead() returns true */
  public static Coinflip of(Scalar p_head) {
    return new Coinflip(p_head);
  }

  // ---
  private final Random random = new Random();
  private final float p_head;

  private Coinflip(Scalar p_head) {
    this.p_head = Clip.unit().requireInside(p_head).number().floatValue();
  }

  /** returns true if the coin toss ended up with head */
  public boolean tossHead() {
    return random.nextFloat() < p_head;
  }
}
