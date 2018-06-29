// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.AnimationWriter;

/** Example 4.2: Jack's Car Rental
 * Figure 4.2
 * 
 * p.87-88 */
enum VI_CarRental {
  ;
  public static void main(String[] args) throws Exception {
    CarRental carRental = new CarRental(20);
    ValueIteration vi = new ValueIteration(carRental);
    try (AnimationWriter gsw = AnimationWriter.of(UserHome.Pictures("carrental_vi.gif"), 250)) {
      for (int count = 0; count <= 25; ++count) {
        System.out.println(count);
        gsw.append(CarRentalHelper.joinAll(carRental, vi.vs()));
        vi.step();
      }
      gsw.append(CarRentalHelper.joinAll(carRental, vi.vs()));
    }
  }
}
